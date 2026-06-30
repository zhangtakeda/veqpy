#pragma once

// Solver workspace, residual callbacks, AD adapters, and solve dispatch for VEQlib kernels.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#ifdef ENABLE_ENZYME
    #include <enzyme/enzyme>
extern int enzyme_const;
extern int enzyme_dupv;
extern int enzyme_width;
#endif
#include <nlohmann/json.hpp>

#include "kernel_case.h"
#include "math.h"
#include "nonlinear.h"
#include "tensor.h"

namespace veqlib_kernel_api
{
    namespace
    {
        using math::is_finite;
        std::array<double, KernelShape::x_size>
        decode_z_to_x(std::span<const double, KernelShape::x_size>   z,
                      std::span<const double, KernelShape::x_size>   x_scale) noexcept
        {
            std::array<double, KernelShape::x_size> x;
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                x[i] = z[i] * x_scale[i];
            return x;
        }

        std::array<double, KernelShape::x_size>
        decode_z_to_x(std::span<const double, KernelShape::x_size>   z,
                      const std::array<double, KernelShape::x_size>& x_scale) noexcept
        {
            return decode_z_to_x(z, std::span<const double, KernelShape::x_size>{x_scale.data(), KernelShape::x_size});
        }

        std::array<double, KernelShape::x_size>
        encode_x_to_z(const std::array<double, KernelShape::x_size>& x,
                      const std::array<double, KernelShape::x_size>& x_scale) noexcept
        {
            std::array<double, KernelShape::x_size> z;
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                z[i] = x[i] / x_scale[i];
            return z;
        }

        KernelOperator::Setup setup_for_case(const CaseInput& input) noexcept
        {
            KernelOperator::Setup setup{};
            setup.profile_params = profile_params_for_case(input);
            for (size_t i = 0; i < KernelSource::sample_count; ++i)
            {
                setup.heat[i]    = input.heat[i];
                setup.current[i] = input.current[i];
            }
            return setup;
        }

        KernelOperator::SolveParams solve_params_for_case(const CaseInput& input) noexcept
        {
            KernelOperator::SolveParams params{};
            params.a  = input.a;
            params.R0 = input.R0;
            params.Z0 = input.Z0;
            params.B0 = input.B0;
            params.Ip = input.Ip;
            params.beta = input.beta;
            return params;
        }

        KernelOperator make_operator_for_case(const CaseInput& input) noexcept
        {
            KernelOperator op{setup_for_case(input)};
            op.set_solve_params(solve_params_for_case(input));
            return op;
        }

        struct SolveContext
        {
            KernelOperator op;
            CaseInput      input{};
            PackedVector   initial_raw{uninitialized};
            PackedVector   initial_scaled{uninitialized};
            std::array<double, 2> initial_alpha{};
            double         initial_raw_norm    = 0.0;
            double         initial_scaled_norm = 0.0;
            bool           has_initial_residual = false;
            int            initial_residual_evaluations = 0;
            int            jacobian_component_evaluations = 0;
            double         residual_callback_ms           = 0.0;
            double         residual_kernel_ms             = 0.0;
            double         residual_scale_ms              = 0.0;
            double         final_residual_ms              = 0.0;
            double         jacobian_callback_ms           = 0.0;
            double         jvp_callback_ms                = 0.0;
            double         linear_solve_ms                = 0.0;

            explicit SolveContext(const CaseInput& case_input)
                : op(make_operator_for_case(case_input)), input(case_input)
            {
            }

            void reset_solve_counters() noexcept
            {
                jacobian_component_evaluations = 0;
                residual_callback_ms           = 0.0;
                residual_kernel_ms             = 0.0;
                residual_scale_ms              = 0.0;
                final_residual_ms              = 0.0;
                jacobian_callback_ms           = 0.0;
                jvp_callback_ms                = 0.0;
                linear_solve_ms                = 0.0;
            }

            void raw_residual(std::span<const double, KernelShape::x_size> x,
                              std::span<double, KernelShape::x_size>       residual) noexcept
            {
                PackedVector raw{uninitialized};
                op.evaluate(x, raw);
                for (size_t i = 0; i < KernelShape::x_size; ++i)
                    residual[i] = raw[i];
            }
        };

        double deterministic_probe_sign(size_t probe, size_t index) noexcept
        {
            uint64_t value = (static_cast<uint64_t>(probe) + 1ULL) * 0x9e3779b97f4a7c15ULL;
            value ^= (static_cast<uint64_t>(index) + 1ULL) * 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 30U;
            value *= 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 27U;
            value *= 0x94d049bb133111ebULL;
            value ^= value >> 31U;
            return (value & 1ULL) == 0ULL ? -1.0 : 1.0;
        }

        std::array<double, KernelShape::x_size> build_safe_residual_scale(SolveContext&       context,
                                                                          const PackedVector& initial_raw,
                                                                          double              floor,
                                                                          double              max_ratio,
                                                                          double              huber_tau,
                                                                          int                 probe_count,
                                                                          double              probe_step,
                                                                          double sensitivity_lambda) noexcept
        {
            if (probe_count <= 0 || !is_finite(probe_step) || probe_step <= 0.0)
                return build_balanced_residual_scale(initial_raw, floor, max_ratio, huber_tau);

            std::array<double, KernelShape::active_count> amplitude_values{};
            size_t                                        offset = 0;
            for (size_t block = 0; block < KernelShape::active_count; ++block)
            {
                const size_t length     = KernelShape::active_lengths[block];
                amplitude_values[block] = robust_rms_block(initial_raw, offset, length, huber_tau);
                offset += length;
            }

            std::array<double, KernelShape::active_count> sensitivity_sq{};
            for (int probe = 0; probe < probe_count; ++probe)
            {
                std::array<double, KernelShape::x_size> probe_x{};
                for (size_t i = 0; i < KernelShape::x_size; ++i)
                    probe_x[i] = context.input.x0[i] + probe_step * context.input.x_scale[i] *
                                                           deterministic_probe_sign(static_cast<size_t>(probe), i);

                PackedVector probe_raw{uninitialized};
                context.raw_residual(std::span<const double, KernelShape::x_size>{probe_x.data(), KernelShape::x_size},
                                     std::span<double, KernelShape::x_size>{probe_raw.data(), KernelShape::x_size});

                PackedVector diff{uninitialized};
                for (size_t i = 0; i < KernelShape::x_size; ++i)
                    diff[i] = (probe_raw[i] - initial_raw[i]) / probe_step;

                offset = 0;
                for (size_t block = 0; block < KernelShape::active_count; ++block)
                {
                    const size_t length = KernelShape::active_lengths[block];
                    const double value  = robust_rms_block(diff, offset, length, huber_tau);
                    sensitivity_sq[block] += value * value;
                    offset += length;
                }
            }

            const double lambda = is_finite(sensitivity_lambda) && sensitivity_lambda > 0.0 ? sensitivity_lambda : 0.0;
            std::array<double, KernelShape::active_count> combined{};
            for (size_t block = 0; block < KernelShape::active_count; ++block)
            {
                const double sensitivity = std::sqrt(sensitivity_sq[block] / static_cast<double>(probe_count));
                combined[block]          = std::hypot(amplitude_values[block], lambda * sensitivity);
            }
            clip_scale_by_anchor(combined, floor, max_ratio);
            return expand_block_scale_values(combined);
        }

        std::array<double, KernelShape::x_size>
        build_residual_scale_for_context(SolveContext& context, const PackedVector& initial_raw) noexcept
        {
            switch (context.input.residual_normalization_code)
            {
            case ResidualNormalizationNone:
                return build_none_residual_scale();
            case ResidualNormalizationFast:
                return build_fast_residual_scale(initial_raw,
                                                 context.input.residual_normalization_floor,
                                                 context.input.residual_normalization_max_ratio);
            case ResidualNormalizationBalanced:
                return build_balanced_residual_scale(initial_raw,
                                                     context.input.residual_normalization_floor,
                                                     context.input.residual_normalization_max_ratio,
                                                     context.input.residual_normalization_huber_tau);
            case ResidualNormalizationSafe:
                return build_safe_residual_scale(context,
                                                 initial_raw,
                                                 context.input.residual_normalization_floor,
                                                 context.input.residual_normalization_max_ratio,
                                                 context.input.residual_normalization_huber_tau,
                                                 context.input.residual_normalization_probe_count,
                                                 context.input.residual_normalization_probe_step,
                                                 context.input.residual_normalization_sensitivity_lambda);
            default:
                return build_none_residual_scale();
            }
        }

        int residual_scale_extra_evaluations(const CaseInput& input) noexcept
        {
            if (input.residual_normalization_code != ResidualNormalizationSafe)
                return 0;
            if (input.residual_normalization_probe_count <= 0 || !is_finite(input.residual_normalization_probe_step) ||
                input.residual_normalization_probe_step <= 0.0)
                return 0;
            return input.residual_normalization_probe_count;
        }

        void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept;

#ifdef ENABLE_ENZYME
        using KernelPlan        = KernelOperator::KernelPlan;
        using KernelSolveParams = KernelOperator::SolveParams;
        using KernelWorkspace   = KernelOperator::KernelWorkspace;

        double scaled_residual_raw_x_for_enzyme(double*                  x,
                                                double*                  fvec,
                                                KernelWorkspace*         workspace,
                                                const KernelPlan*        plan,
                                                const KernelSolveParams* solve_params,
                                                const double*            residual_scale) noexcept
        {
            PackedVector raw{uninitialized};
            KernelOperator::evaluate_with(*plan,
                                          *solve_params,
                                          *workspace,
                                          std::span<const double, KernelShape::x_size>{x, KernelShape::x_size},
                                          raw);
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                fvec[i] = raw[i] / residual_scale[i];
            return 0.0;
        }
#endif

        void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept
        {
            const auto   callback_started = std::chrono::steady_clock::now();
            const auto   x = decode_z_to_x(std::span<const double, KernelShape::x_size>{z, KernelShape::x_size},
                                         context.input.x_scale);
            PackedVector raw{uninitialized};
            const auto   kernel_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, KernelShape::x_size>{x.data(), KernelShape::x_size},
                                 std::span<double, KernelShape::x_size>{raw.data(), KernelShape::x_size});
            context.residual_kernel_ms += elapsed_ms_since(kernel_started);
            const auto scale_started = std::chrono::steady_clock::now();
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                fvec[i] = raw[i] / context.input.residual_scale[i];
            context.residual_scale_ms += elapsed_ms_since(scale_started);
            context.residual_callback_ms += elapsed_ms_since(callback_started);
        }

#ifdef ENABLE_ENZYME
        void fill_enzyme_jvp_z(SolveContext& context, const double* z, const double* v, double* jv)
        {
            std::array<double, KernelShape::x_size> x_primal;
            std::array<double, KernelShape::x_size> x_dot;
            std::array<double, KernelShape::x_size> f_primal;
            const auto&                             x_scale = context.input.x_scale;
            for (size_t i = 0; i < KernelShape::x_size; ++i)
            {
                x_primal[i] = z[i] * x_scale[i];
                x_dot[i]    = v[i] * x_scale[i];
            }

            KernelWorkspace jvp_workspace_dot{};
            const KernelPlan&        plan         = context.op.plan;
            const KernelSolveParams& solve_params = context.op.solve_params();
            (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
                scaled_residual_raw_x_for_enzyme,
                enzyme::Duplicated<double*>{x_primal.data(), x_dot.data()},
                enzyme::Duplicated<double*>{f_primal.data(), jv},
                enzyme::Duplicated<KernelWorkspace*>{&context.op.workspace, &jvp_workspace_dot},
                enzyme::Const<const KernelPlan*>{&plan},
                enzyme::Const<const KernelSolveParams*>{&solve_params},
                enzyme::Const<const double*>{context.input.residual_scale.data()});
        }

        constexpr size_t enzyme_dense_jacobian_batch_width() noexcept
        {
            if constexpr (Topology::enzyme_jacobian_batch_width > 0)
                return Topology::enzyme_jacobian_batch_width;
            if constexpr (KernelShape::x_size >= 8)
                return 4;
            else
                return 1;
        }

        template <size_t Width>
        void fill_enzyme_jacobian_z_vector(SolveContext& context, const double* z, double* fjac, int ldfjac)
        {
            static_assert(Width > 0);
            constexpr size_t n                 = KernelShape::x_size;
            constexpr size_t lane_stride_bytes = n * sizeof(double);

            std::array<double, n> x_primal;
            const auto&           x_scale = context.input.x_scale;
            for (size_t i = 0; i < n; ++i)
                x_primal[i] = z[i] * x_scale[i];

            const KernelPlan&        plan         = context.op.plan;
            const KernelSolveParams& solve_params = context.op.solve_params();

            for (size_t first_col = 0; first_col < n; first_col += Width)
            {
                std::array<double, Width * n>     x_dot{};
                std::array<double, n>             f_primal;
                std::array<double, Width * n>     f_dot;
                std::array<KernelWorkspace, Width> chunk_workspace_dot{};

                const size_t lane_count = std::min(Width, n - first_col);
                for (size_t lane = 0; lane < lane_count; ++lane)
                    x_dot[lane * n + first_col + lane] = x_scale[first_col + lane];

                __enzyme_fwddiff<void>(reinterpret_cast<void*>(scaled_residual_raw_x_for_enzyme),
                                       enzyme_width,
                                       static_cast<int>(Width),
                                       enzyme_dupv,
                                       static_cast<int>(lane_stride_bytes),
                                       x_primal.data(),
                                       x_dot.data(),
                                       enzyme_dupv,
                                       static_cast<int>(lane_stride_bytes),
                                       f_primal.data(),
                                       f_dot.data(),
                                       enzyme_dupv,
                                       static_cast<int>(sizeof(KernelWorkspace)),
                                       &context.op.workspace,
                                       chunk_workspace_dot.data(),
                                       enzyme_const,
                                       &plan,
                                       enzyme_const,
                                       &solve_params,
                                       enzyme_const,
                                       context.input.residual_scale.data());

                for (size_t lane = 0; lane < lane_count; ++lane)
                {
                    const size_t col = first_col + lane;
                    for (size_t row = 0; row < n; ++row)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[lane * n + row];
                }
            }
            context.jacobian_component_evaluations += static_cast<int>(n);
        }

        void fill_enzyme_jacobian_z_scalar(SolveContext& context, const double* z, double* fjac, int ldfjac)
        {
            std::array<double, KernelShape::x_size> x_primal;
            const auto&                             x_scale = context.input.x_scale;
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                x_primal[i] = z[i] * x_scale[i];

            const KernelPlan&        plan         = context.op.plan;
            const KernelSolveParams& solve_params = context.op.solve_params();

            for (size_t col = 0; col < KernelShape::x_size; ++col)
            {
                std::array<double, KernelShape::x_size> x_dot{};
                std::array<double, KernelShape::x_size> f_primal;
                std::array<double, KernelShape::x_size> f_dot;
                KernelWorkspace                         column_workspace_dot{};
                x_dot[col] = x_scale[col];
                (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
                    scaled_residual_raw_x_for_enzyme,
                    enzyme::Duplicated<double*>{x_primal.data(), x_dot.data()},
                    enzyme::Duplicated<double*>{f_primal.data(), f_dot.data()},
                    enzyme::Duplicated<KernelWorkspace*>{&context.op.workspace, &column_workspace_dot},
                    enzyme::Const<const KernelPlan*>{&plan},
                    enzyme::Const<const KernelSolveParams*>{&solve_params},
                    enzyme::Const<const double*>{context.input.residual_scale.data()});
                for (size_t row = 0; row < KernelShape::x_size; ++row)
                    fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[row];
            }
            context.jacobian_component_evaluations += static_cast<int>(KernelShape::x_size);
        }

        void fill_enzyme_jacobian_z(SolveContext& context, const double* z, double* fjac, int ldfjac)
        {
            constexpr size_t width = enzyme_dense_jacobian_batch_width();
            if constexpr (width == 1)
                fill_enzyme_jacobian_z_scalar(context, z, fjac, ldfjac);
            else
                fill_enzyme_jacobian_z_vector<width>(context, z, fjac, ldfjac);
        }
#endif

        struct ScaledResidualProblem
        {
            static constexpr size_t equations = KernelShape::x_size;
            static constexpr size_t variables = KernelShape::x_size;

            SolveContext* context = nullptr;

            void operator()(const double* z, double* fvec) const noexcept
            {
                (void)scaled_residual_z_no_count(*context, z, fvec);
            }

#ifdef ENABLE_ENZYME
            void jacobian(const double* z, double* jacobian) const
            {
                constexpr size_t          n = KernelShape::x_size;
                std::array<double, n * n> column_major{};
                const auto                started = std::chrono::steady_clock::now();
                fill_enzyme_jacobian_z(*context, z, column_major.data(), static_cast<int>(n));
                context->jacobian_callback_ms += elapsed_ms_since(started);
                for (size_t row = 0; row < n; ++row)
                    for (size_t col = 0; col < n; ++col)
                        jacobian[row * n + col] = column_major[row + n * col];
            }

            void jvp(const double* z, const double* v, double* jv) const
            {
                const auto started = std::chrono::steady_clock::now();
                fill_enzyme_jvp_z(*context, z, v, jv);
                context->jvp_callback_ms += elapsed_ms_since(started);
            }
#endif
        };

        template <typename Context>
        int jacobian_evaluation_count(const Context& context) noexcept
        {
            if constexpr (requires { context.jacobian_evaluations; })
                return context.jacobian_evaluations;
            else
                return 0;
        }

        template <typename Context>
        int jvp_evaluation_count(const Context& context) noexcept
        {
            if constexpr (requires { context.jvp_evaluations; })
                return context.jvp_evaluations;
            else
                return 0;
        }

        template <typename Context>
        int linear_iteration_count(const Context& context) noexcept
        {
            if constexpr (requires { context.linear_iterations; })
                return context.linear_iterations;
            else
                return 0;
        }

        void fill_solve_result_from_z(
            SolveContext& context, SolveResult& result, const double* z, int info, int nfev, int njev, int callbacks)
        {
            result.info      = info;
            result.nfev      = context.initial_residual_evaluations + nfev + 1;
            result.njev      = njev;
            result.callbacks = callbacks;
            result.solver_nfev = nfev;
            result.x         = decode_z_to_x(std::span<const double, KernelShape::x_size>{z, KernelShape::x_size},
                                     context.input.x_scale);
            const auto final_residual_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, KernelShape::x_size>{result.x.data(), KernelShape::x_size},
                                 std::span<double, KernelShape::x_size>{result.raw.data(), KernelShape::x_size});
            result.final_residual_ms = elapsed_ms_since(final_residual_started);
            for (size_t i = 0; i < KernelShape::x_size; ++i)
                result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
            result.raw_norm             = norm2(std::span<const double, KernelShape::x_size>{
                result.raw.data(),
                KernelShape::x_size,
            });
            result.scaled_norm          = norm2(std::span<const double, KernelShape::x_size>{
                result.scaled.data(),
                KernelShape::x_size,
            });
            result.residual_callback_ms = context.residual_callback_ms;
            result.residual_kernel_ms   = context.residual_kernel_ms;
            result.residual_scale_ms    = context.residual_scale_ms;
            result.jacobian_callback_ms = context.jacobian_callback_ms;
            result.jvp_callback_ms      = context.jvp_callback_ms;
            result.linear_solve_ms      = context.linear_solve_ms;
            result.alpha    = {context.op.workspace.source_runtime.alpha1, context.op.workspace.source_runtime.alpha2};
            result.accepted = result.raw_norm <= acceptance_threshold(context.input);
            result.cert_threshold                     = acceptance_threshold(context.input);
            result.initial_raw_norm                   = context.initial_raw_norm;
            result.fast_path_raw_norm                 = context.initial_raw_norm;
            result.initial_residual_evaluations       = context.initial_residual_evaluations;
            result.certification_residual_evaluations = 0;
            result.total_raw_residual_evaluations     = result.nfev;
            result.accepted_by                        = "solver";
            result.fast_path                          = "none";
            result.fallback_used                      = false;
            result.fallback_reason                    = "";
        }

        template <typename SolverContext>
        void configure_common_solver(SolverContext& solver_context, const CaseInput& input) noexcept
        {
            if constexpr (requires { solver_context.tolerance; })
                solver_context.tolerance = input.max_residual;
            if constexpr (requires { solver_context.max_evaluations; })
                solver_context.max_evaluations = max_solver_evaluations(input);
            if constexpr (requires { solver_context.max_iterations; })
                solver_context.max_iterations = max_solver_evaluations(input);
            if constexpr (requires { solver_context.max_dimension; })
                solver_context.max_dimension = static_cast<int>(KernelShape::x_size);
        }

        template <typename SolverContext>
        void configure_scaled_z_solver(SolverContext& solver_context, const CaseInput& input) noexcept
        {
            configure_common_solver(solver_context, input);
            if constexpr (requires { solver_context.finite_difference_step; })
                solver_context.finite_difference_step = veqpy_hybr_eps;
            if constexpr (requires { solver_context.initial_step_bound; })
                solver_context.initial_step_bound = veqpy_hybr_factor;
            if constexpr (requires { solver_context.lower_bandwidth; })
                solver_context.lower_bandwidth = static_cast<int>(KernelShape::x_size) - 1;
            if constexpr (requires { solver_context.upper_bandwidth; })
                solver_context.upper_bandwidth = static_cast<int>(KernelShape::x_size) - 1;
            if constexpr (requires { solver_context.scale_mode; })
                solver_context.scale_mode = veqpy_hybr_mode;
            if constexpr (requires { solver_context.print_interval; })
                solver_context.print_interval = veqpy_hybr_nprint;
        }

        template <typename SolverContext>
        void configure_levenberg_marquardt_solver(SolverContext& solver_context, const CaseInput& input) noexcept
        {
            configure_common_solver(solver_context, input);
            if constexpr (requires { solver_context.finite_difference_step; })
                solver_context.finite_difference_step = veqpy_lm_eps;
            if constexpr (requires { solver_context.initial_step_bound; })
                solver_context.initial_step_bound = veqpy_lm_factor;
            if constexpr (requires { solver_context.scale_mode; })
                solver_context.scale_mode = veqpy_lm_mode;
            if constexpr (requires { solver_context.print_interval; })
                solver_context.print_interval = veqpy_lm_nprint;
        }

        template <typename Policy>
        SolveResult run_nonlinear_policy_once(SolveContext& context)
        {
            context.reset_solve_counters();
            const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, KernelShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualProblem problem{&context};
            auto                  solver = nonlinear::make_solver<Policy>(problem);
            configure_scaled_z_solver(solver.context, context.input);

            solver.optimize_inplace(z);

            SolveResult result{};
            fill_solve_result_from_z(context,
                                     result,
                                     z.data(),
                                     solver.context.info,
                                     solver.context.evaluations,
                                     jacobian_evaluation_count(solver.context),
                                     solver.context.evaluations);
            result.jacobian_component_evaluations = context.jacobian_component_evaluations;
            result.jvp_evaluations                = jvp_evaluation_count(solver.context);
            result.linear_iterations              = linear_iteration_count(solver.context);
            return result;
        }

        SolveResult run_levenberg_marquardt_once(SolveContext& context)
        {
            context.reset_solve_counters();
            const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, KernelShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualProblem problem{&context};
            auto                  solver = nonlinear::make_solver<nonlinear::LevenbergMarquardt>(problem);
            configure_levenberg_marquardt_solver(solver.context, context.input);

            solver.optimize_inplace(z);

            SolveResult result{};
            fill_solve_result_from_z(context,
                                     result,
                                     z.data(),
                                     solver.context.info,
                                     solver.context.evaluations,
                                     jacobian_evaluation_count(solver.context),
                                     solver.context.evaluations);
            result.jacobian_component_evaluations = context.jacobian_component_evaluations;
            result.jvp_evaluations                = jvp_evaluation_count(solver.context);
            result.linear_iterations              = linear_iteration_count(solver.context);
            return result;
        }

        SolveResult run_solver_once(SolveContext& context)
        {
            if (context.input.solver == SolverKind::LevenbergMarquardt)
                return run_levenberg_marquardt_once(context);
            if (context.input.solver == SolverKind::NewtonKrylov)
                return run_nonlinear_policy_once<nonlinear::NewtonKrylov>(context);
            if (context.input.solver == SolverKind::NewtonRaphson)
                return run_nonlinear_policy_once<nonlinear::NewtonRaphson>(context);
            if (context.input.solver == SolverKind::Powell)
                return run_nonlinear_policy_once<nonlinear::Powell>(context);
            throw std::runtime_error("unsupported solver kind");
        }

        nlohmann::json solve_result_json(const SolveResult& result)
        {
            return {
                {"accepted_by_veqpy", result.accepted},
                {"x", json_array(result.x)},
                {"raw_residual", json_array(result.raw)},
                {"scaled_residual", json_array(result.scaled)},
                {"alpha", json_array(result.alpha)},
                {"raw_norm", result.raw_norm},
                {"scaled_norm", result.scaled_norm},
                {"info", result.info},
                {"nfev", result.nfev},
                {"solver_nfev", result.solver_nfev},
                {"njev", result.njev},
                {"callback_evaluations", result.callbacks},
                {"jacobian_component_evaluations", result.jacobian_component_evaluations},
                {"jvp_evaluations", result.jvp_evaluations},
                {"linear_iterations", result.linear_iterations},
                {"accepted_by", result.accepted_by},
                {"fast_path", result.fast_path},
                {"fallback_used", result.fallback_used},
                {"fallback_reason", result.fallback_reason},
                {"cert_threshold", result.cert_threshold},
                {"initial_raw_norm", result.initial_raw_norm},
                {"fast_path_raw_norm", result.fast_path_raw_norm},
                {"initial_residual_evaluations", result.initial_residual_evaluations},
                {"certification_residual_evaluations", result.certification_residual_evaluations},
                {"total_raw_residual_evaluations", result.total_raw_residual_evaluations},
                {"callback_timing_ms",
                 {
                     {"residual_total", result.residual_callback_ms},
                     {"residual_kernel", result.residual_kernel_ms},
                     {"residual_scale", result.residual_scale_ms},
                     {"final_residual", result.final_residual_ms},
                     {"jacobian_total", result.jacobian_callback_ms},
                     {"jvp_total", result.jvp_callback_ms},
                     {"linear_solve", result.linear_solve_ms},
                 }},
            };
        }

    } // namespace

} // namespace veqlib_kernel_api
