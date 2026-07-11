#pragma once

// Solver workspace, residual callbacks, AD adapters, and solve dispatch for generated Cxx Kernel artifacts.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
    #include <chrono>
#endif

#ifdef ENABLE_ENZYME
    #include <enzyme/enzyme>
extern int enzyme_const;
extern int enzyme_dupv;
extern int enzyme_width;
#endif

#include "kernel_case.h"
#include "math.h"
#include "nonlinear.h"
#include "tensor.h"

namespace cxx_kernel_api
{
    namespace
    {
        using math::is_finite;
        std::array<double, CompiledShape::x_size>
        decode_z_to_x(std::span<const double, CompiledShape::x_size>   z,
                      std::span<const double, CompiledShape::x_size>   x_scale) noexcept
        {
            std::array<double, CompiledShape::x_size> x;
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                x[i] = z[i] * x_scale[i];
            return x;
        }

        std::array<double, CompiledShape::x_size>
        decode_z_to_x(std::span<const double, CompiledShape::x_size>   z,
                      const std::array<double, CompiledShape::x_size>& x_scale) noexcept
        {
            return decode_z_to_x(z, std::span<const double, CompiledShape::x_size>{x_scale.data(), CompiledShape::x_size});
        }

        std::array<double, CompiledShape::x_size>
        encode_x_to_z(const std::array<double, CompiledShape::x_size>& x,
                      const std::array<double, CompiledShape::x_size>& x_scale) noexcept
        {
            std::array<double, CompiledShape::x_size> z;
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                z[i] = x[i] / x_scale[i];
            return z;
        }

        CompiledOperator::Setup setup_for_case(const RuntimeCase& input) noexcept
        {
            CompiledOperator::Setup setup{};
            setup.profile_params = profile_params_for_case(input);
            for (size_t i = 0; i < CompiledSource::sample_count; ++i)
            {
                setup.heat[i]    = input.heat[i];
                setup.current[i] = input.current[i];
            }
            return setup;
        }

        CompiledOperator::RuntimeScalars runtime_scalars_for_case(const RuntimeCase& input) noexcept
        {
            CompiledOperator::RuntimeScalars params{};
            params.a  = input.a;
            params.R0 = input.R0;
            params.Z0 = input.Z0;
            params.B0 = input.B0;
            params.Ip = input.Ip;
            params.beta = input.beta;
            return params;
        }

        CompiledOperator make_operator_for_case(const RuntimeCase& input) noexcept
        {
            CompiledOperator op{setup_for_case(input)};
            op.set_runtime_scalars(runtime_scalars_for_case(input));
            return op;
        }

        struct SolveState
        {
            CompiledOperator op;
            RuntimeCase      input{};
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
            nonlinear::Workspace<CompiledShape::x_size>& nonlinear_workspace;

            SolveState(const RuntimeCase&                                case_input,
                       nonlinear::Workspace<CompiledShape::x_size>& workspace)
                : op(make_operator_for_case(case_input)), input(case_input), nonlinear_workspace(workspace)
            {
            }

            void reset_case(const RuntimeCase& case_input) noexcept
            {
                input = case_input;
                op.reprepare(setup_for_case(input));
                op.set_runtime_scalars(runtime_scalars_for_case(input));
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

            void raw_residual(std::span<const double, CompiledShape::x_size> x,
                              std::span<double, CompiledShape::x_size>       residual) noexcept
            {
                PackedVector raw{uninitialized};
                op.evaluate(x, raw);
                for (size_t i = 0; i < CompiledShape::x_size; ++i)
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

        std::array<double, CompiledShape::x_size> build_safe_residual_scale(SolveState&       context,
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

            std::array<double, CompiledShape::active_count> amplitude_values{};
            size_t                                        offset = 0;
            for (size_t block = 0; block < CompiledShape::active_count; ++block)
            {
                const size_t length     = CompiledShape::active_lengths[block];
                amplitude_values[block] = robust_rms_block(initial_raw, offset, length, huber_tau);
                offset += length;
            }

            std::array<double, CompiledShape::active_count> sensitivity_sq{};
            for (int probe = 0; probe < probe_count; ++probe)
            {
                std::array<double, CompiledShape::x_size> probe_x{};
                for (size_t i = 0; i < CompiledShape::x_size; ++i)
                    probe_x[i] = context.input.x0[i] + probe_step * context.input.x_scale[i] *
                                                           deterministic_probe_sign(static_cast<size_t>(probe), i);

                PackedVector probe_raw{uninitialized};
                context.raw_residual(std::span<const double, CompiledShape::x_size>{probe_x.data(), CompiledShape::x_size},
                                     std::span<double, CompiledShape::x_size>{probe_raw.data(), CompiledShape::x_size});

                PackedVector diff{uninitialized};
                for (size_t i = 0; i < CompiledShape::x_size; ++i)
                    diff[i] = (probe_raw[i] - initial_raw[i]) / probe_step;

                offset = 0;
                for (size_t block = 0; block < CompiledShape::active_count; ++block)
                {
                    const size_t length = CompiledShape::active_lengths[block];
                    const double value  = robust_rms_block(diff, offset, length, huber_tau);
                    sensitivity_sq[block] += value * value;
                    offset += length;
                }
            }

            const double lambda = is_finite(sensitivity_lambda) && sensitivity_lambda > 0.0 ? sensitivity_lambda : 0.0;
            std::array<double, CompiledShape::active_count> combined{};
            for (size_t block = 0; block < CompiledShape::active_count; ++block)
            {
                const double sensitivity = std::sqrt(sensitivity_sq[block] / static_cast<double>(probe_count));
                combined[block]          = std::hypot(amplitude_values[block], lambda * sensitivity);
            }
            clip_scale_by_anchor(combined, floor, max_ratio);
            return expand_block_scale_values(combined);
        }

        std::array<double, CompiledShape::x_size>
        build_residual_scale_for_context(SolveState& context, const PackedVector& initial_raw) noexcept
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

        int residual_scale_extra_evaluations(const RuntimeCase& input) noexcept
        {
            if (input.residual_normalization_code != ResidualNormalizationSafe)
                return 0;
            if (input.residual_normalization_probe_count <= 0 || !is_finite(input.residual_normalization_probe_step) ||
                input.residual_normalization_probe_step <= 0.0)
                return 0;
            return input.residual_normalization_probe_count;
        }

        void scaled_residual_z_no_count(SolveState& context, const double* z, double* fvec) noexcept;

#ifdef ENABLE_ENZYME
        using OperatorPlan        = CompiledOperator::OperatorPlan;
        using RuntimeScalars = CompiledOperator::RuntimeScalars;
        using OperatorWorkspace   = CompiledOperator::OperatorWorkspace;

        double scaled_residual_raw_x_for_enzyme(double*                  x,
                                                double*                  fvec,
                                                OperatorWorkspace*         workspace,
                                                const OperatorPlan*        plan,
                                                const RuntimeScalars* runtime_scalars,
                                                const double*            residual_scale) noexcept
        {
            PackedVector raw{uninitialized};
            CompiledOperator::evaluate_with(*plan,
                                          *runtime_scalars,
                                          *workspace,
                                          std::span<const double, CompiledShape::x_size>{x, CompiledShape::x_size},
                                          raw);
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                fvec[i] = raw[i] / residual_scale[i];
            return 0.0;
        }
#endif

        void scaled_residual_z_no_count(SolveState& context, const double* z, double* fvec) noexcept
        {
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            const auto   callback_started = std::chrono::steady_clock::now();
#endif
            const auto   x = decode_z_to_x(std::span<const double, CompiledShape::x_size>{z, CompiledShape::x_size},
                                         context.input.x_scale);
            PackedVector raw{uninitialized};
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            const auto   kernel_started = std::chrono::steady_clock::now();
#endif
            context.raw_residual(std::span<const double, CompiledShape::x_size>{x.data(), CompiledShape::x_size},
                                 std::span<double, CompiledShape::x_size>{raw.data(), CompiledShape::x_size});
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            context.residual_kernel_ms += elapsed_ms_since(kernel_started);
            const auto scale_started = std::chrono::steady_clock::now();
#endif
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                fvec[i] = raw[i] / context.input.residual_scale[i];
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            context.residual_scale_ms += elapsed_ms_since(scale_started);
            context.residual_callback_ms += elapsed_ms_since(callback_started);
#endif
        }

#ifdef ENABLE_ENZYME
        void fill_enzyme_jvp_z(SolveState& context, const double* z, const double* v, double* jv)
        {
            std::array<double, CompiledShape::x_size> x_primal;
            std::array<double, CompiledShape::x_size> x_dot;
            std::array<double, CompiledShape::x_size> f_primal;
            const auto&                             x_scale = context.input.x_scale;
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
            {
                x_primal[i] = z[i] * x_scale[i];
                x_dot[i]    = v[i] * x_scale[i];
            }

            OperatorWorkspace jvp_workspace_dot{};
            const OperatorPlan&        plan         = context.op.plan;
            const RuntimeScalars& runtime_scalars = context.op.runtime_scalars();
            (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
                scaled_residual_raw_x_for_enzyme,
                enzyme::Duplicated<double*>{x_primal.data(), x_dot.data()},
                enzyme::Duplicated<double*>{f_primal.data(), jv},
                enzyme::Duplicated<OperatorWorkspace*>{&context.op.workspace, &jvp_workspace_dot},
                enzyme::Const<const OperatorPlan*>{&plan},
                enzyme::Const<const RuntimeScalars*>{&runtime_scalars},
                enzyme::Const<const double*>{context.input.residual_scale.data()});
        }

        constexpr size_t enzyme_dense_jacobian_batch_width() noexcept
        {
            if constexpr (Topology::enzyme_jacobian_batch_width > 0)
                return Topology::enzyme_jacobian_batch_width;
            if constexpr (CompiledShape::x_size >= 8)
                return 4;
            else
                return 1;
        }

        template <size_t Width>
        void fill_enzyme_jacobian_z_vector(SolveState& context, const double* z, double* fjac, int ldfjac)
        {
            static_assert(Width > 0);
            constexpr size_t n                 = CompiledShape::x_size;
            constexpr size_t lane_stride_bytes = n * sizeof(double);

            std::array<double, n> x_primal;
            const auto&           x_scale = context.input.x_scale;
            for (size_t i = 0; i < n; ++i)
                x_primal[i] = z[i] * x_scale[i];

            const OperatorPlan&        plan         = context.op.plan;
            const RuntimeScalars& runtime_scalars = context.op.runtime_scalars();

            for (size_t first_col = 0; first_col < n; first_col += Width)
            {
                std::array<double, Width * n>     x_dot{};
                std::array<double, n>             f_primal;
                std::array<double, Width * n>     f_dot;
                std::array<OperatorWorkspace, Width> chunk_workspace_dot{};

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
                                       static_cast<int>(sizeof(OperatorWorkspace)),
                                       &context.op.workspace,
                                       chunk_workspace_dot.data(),
                                       enzyme_const,
                                       &plan,
                                       enzyme_const,
                                       &runtime_scalars,
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

        void fill_enzyme_jacobian_z_scalar(SolveState& context, const double* z, double* fjac, int ldfjac)
        {
            std::array<double, CompiledShape::x_size> x_primal;
            const auto&                             x_scale = context.input.x_scale;
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                x_primal[i] = z[i] * x_scale[i];

            const OperatorPlan&        plan         = context.op.plan;
            const RuntimeScalars& runtime_scalars = context.op.runtime_scalars();

            for (size_t col = 0; col < CompiledShape::x_size; ++col)
            {
                std::array<double, CompiledShape::x_size> x_dot{};
                std::array<double, CompiledShape::x_size> f_primal;
                std::array<double, CompiledShape::x_size> f_dot;
                OperatorWorkspace                         column_workspace_dot{};
                x_dot[col] = x_scale[col];
                (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
                    scaled_residual_raw_x_for_enzyme,
                    enzyme::Duplicated<double*>{x_primal.data(), x_dot.data()},
                    enzyme::Duplicated<double*>{f_primal.data(), f_dot.data()},
                    enzyme::Duplicated<OperatorWorkspace*>{&context.op.workspace, &column_workspace_dot},
                    enzyme::Const<const OperatorPlan*>{&plan},
                    enzyme::Const<const RuntimeScalars*>{&runtime_scalars},
                    enzyme::Const<const double*>{context.input.residual_scale.data()});
                for (size_t row = 0; row < CompiledShape::x_size; ++row)
                    fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[row];
            }
            context.jacobian_component_evaluations += static_cast<int>(CompiledShape::x_size);
        }

        void fill_enzyme_jacobian_z(SolveState& context, const double* z, double* fjac, int ldfjac)
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
            static constexpr size_t equations = CompiledShape::x_size;
            static constexpr size_t variables = CompiledShape::x_size;

            SolveState* context = nullptr;

            void operator()(const double* z, double* fvec) const noexcept
            {
                (void)scaled_residual_z_no_count(*context, z, fvec);
            }

#ifdef ENABLE_ENZYME
            void jacobian(const double* z, double* jacobian) const
            {
                constexpr size_t          n = CompiledShape::x_size;
                std::array<double, n * n> column_major{};
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
                const auto                started = std::chrono::steady_clock::now();
#endif
                fill_enzyme_jacobian_z(*context, z, column_major.data(), static_cast<int>(n));
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
                context->jacobian_callback_ms += elapsed_ms_since(started);
#endif
                for (size_t row = 0; row < n; ++row)
                    for (size_t col = 0; col < n; ++col)
                        jacobian[row * n + col] = column_major[row + n * col];
            }

            void jvp(const double* z, const double* v, double* jv) const
            {
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
                const auto started = std::chrono::steady_clock::now();
#endif
                fill_enzyme_jvp_z(*context, z, v, jv);
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
                context->jvp_callback_ms += elapsed_ms_since(started);
#endif
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
            SolveState& context, SolveResult& result, const double* z, int info, int nfev, int njev, int callbacks)
        {
            result.info      = info;
            result.nfev      = context.initial_residual_evaluations + nfev + 1;
            result.njev      = njev;
            result.callbacks = callbacks;
            result.solver_nfev = nfev;
            result.x         = decode_z_to_x(std::span<const double, CompiledShape::x_size>{z, CompiledShape::x_size},
                                     context.input.x_scale);
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            const auto final_residual_started = std::chrono::steady_clock::now();
#endif
            context.raw_residual(std::span<const double, CompiledShape::x_size>{result.x.data(), CompiledShape::x_size},
                                 std::span<double, CompiledShape::x_size>{result.raw.data(), CompiledShape::x_size});
#if defined(VEQPY_CXX_DETAILED_SOLVE_TIMING)
            result.final_residual_ms = elapsed_ms_since(final_residual_started);
#endif
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
            result.raw_norm             = norm2(std::span<const double, CompiledShape::x_size>{
                result.raw.data(),
                CompiledShape::x_size,
            });
            result.scaled_norm          = norm2(std::span<const double, CompiledShape::x_size>{
                result.scaled.data(),
                CompiledShape::x_size,
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
        void configure_common_solver(SolverContext& solver_context, const RuntimeCase& input) noexcept
        {
            if constexpr (requires { solver_context.tolerance; })
                solver_context.tolerance = input.max_residual;
            if constexpr (requires { solver_context.max_evaluations; })
                solver_context.max_evaluations = max_solver_evaluations(input);
            if constexpr (requires { solver_context.max_iterations; })
                solver_context.max_iterations = max_solver_evaluations(input);
            if constexpr (requires { solver_context.max_dimension; })
                solver_context.max_dimension = static_cast<int>(CompiledShape::x_size);
        }

        template <typename SolverContext>
        void configure_scaled_z_solver(SolverContext& solver_context, const RuntimeCase& input) noexcept
        {
            configure_common_solver(solver_context, input);
            if constexpr (requires { solver_context.finite_difference_step; })
                solver_context.finite_difference_step = default_hybr_eps;
            if constexpr (requires { solver_context.initial_step_bound; })
                solver_context.initial_step_bound = default_hybr_factor;
            if constexpr (requires { solver_context.lower_bandwidth; })
                solver_context.lower_bandwidth = static_cast<int>(CompiledShape::x_size) - 1;
            if constexpr (requires { solver_context.upper_bandwidth; })
                solver_context.upper_bandwidth = static_cast<int>(CompiledShape::x_size) - 1;
            if constexpr (requires { solver_context.scale_mode; })
                solver_context.scale_mode = default_hybr_mode;
            if constexpr (requires { solver_context.print_interval; })
                solver_context.print_interval = default_hybr_nprint;
        }

        template <typename SolverContext>
        void configure_levenberg_marquardt_solver(SolverContext& solver_context, const RuntimeCase& input) noexcept
        {
            configure_common_solver(solver_context, input);
            if constexpr (requires { solver_context.finite_difference_step; })
                solver_context.finite_difference_step = default_lm_eps;
            if constexpr (requires { solver_context.initial_step_bound; })
                solver_context.initial_step_bound = default_lm_factor;
            if constexpr (requires { solver_context.scale_mode; })
                solver_context.scale_mode = default_lm_mode;
            if constexpr (requires { solver_context.print_interval; })
                solver_context.print_interval = default_lm_nprint;
        }

        template <typename Policy>
        SolveResult run_nonlinear_policy_once(SolveState& context)
        {
            context.reset_solve_counters();
            const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, CompiledShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualProblem problem{&context};
            auto                  solver = nonlinear::make_solver<Policy>(problem, context.nonlinear_workspace);
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

        SolveResult run_levenberg_marquardt_once(SolveState& context)
        {
            context.reset_solve_counters();
            const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, CompiledShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualProblem problem{&context};
            auto                  solver =
                nonlinear::make_solver<nonlinear::LevenbergMarquardt>(problem, context.nonlinear_workspace);
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

        SolveResult run_solver_once(SolveState& context)
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

    } // namespace

} // namespace cxx_kernel_api
