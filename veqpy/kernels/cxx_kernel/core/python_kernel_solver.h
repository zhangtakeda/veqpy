#pragma once

// Nanobind-facing NativeSolver implementation for generated Cxx Kernel artifacts.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "kernel_case.h"
#include "kernel_runtime.h"
#include "kernel_topology.h"
#include "linalg.h"
#include "veq_numeric.h"
#include "tensor.h"

namespace nb = nanobind;

namespace cxx_python
{
    using std::size_t;

    using config::Topology;
    using cxx_kernel_api::CompiledGrid;
    using cxx_kernel_api::CompiledShape;
    using cxx_kernel_api::CompiledSource;
    using cxx_kernel_api::RuntimeCase;
    using cxx_kernel_api::InitialPolicyColdGeometric;
    using cxx_kernel_api::InitialPolicyCold;
    using cxx_kernel_api::ContinuePolicyColdZeros;
    using cxx_kernel_api::ContinuePolicyColdGeometric;
    using cxx_kernel_api::ContinuePolicyCold;
    using cxx_kernel_api::ContinuePolicyWarmFixed;
    using cxx_kernel_api::ContinuePolicyWarmPredict;
    using cxx_kernel_api::ContinuePolicyWarmChord;
    using cxx_kernel_api::PackedVector;
    using cxx_kernel_api::SolveState;
    using cxx_kernel_api::SolveResult;
    using cxx_kernel_api::SolverKind;
    using cxx_kernel_api::build_residual_scale_for_context;
    using cxx_kernel_api::build_x_block_scale_vector;
    using cxx_kernel_api::build_inline_case;
    using cxx_kernel_api::apply_cold_policy;
    using cxx_kernel_api::apply_initial_policy;
    using cxx_kernel_api::boundary_curve_strain;
    using cxx_kernel_api::norm2;
    using cxx_kernel_api::residual_scale_extra_evaluations;
    using cxx_kernel_api::profile_params_for_case;
    using cxx_kernel_api::run_solver_once;
    using cxx_kernel_api::solver_entrypoint;
    using cxx_kernel_api::solver_info_succeeded;
    using cxx_kernel_api::solver_jacobian;
    using cxx_kernel_api::solver_kind_from_runtime_method_code;
    using cxx_kernel_api::solver_method_code;
    using cxx_kernel_api::solver_method;
    using cxx_kernel_api::initial_policy_name;
    using cxx_kernel_api::continue_policy_is_cold;
    using cxx_kernel_api::continue_policy_name;
    using cxx_kernel_api::continue_policy_uses_chord;
    using cxx_kernel_api::continue_policy_uses_predictor;
    using cxx_kernel_api::continue_policy_uses_warm_state;
    using cxx_kernel_api::residual_normalization_name;
    using cxx_kernel_api::resolved_continue_policy;
    using cxx_kernel_api::validate_initial_policy_code;
    using cxx_kernel_api::validate_continue_policy_code;
    using cxx_kernel_api::validate_residual_normalization_code;
    using cxx_kernel_api::kernel_c_counts;
    using cxx_kernel_api::kernel_s_counts;
#ifdef ENABLE_ENZYME
    using cxx_kernel_api::enzyme_dense_jacobian_batch_width;
#endif

    using tensor::uninitialized;

    using PackedArrayView          = nb::ndarray<nb::numpy, const double, nb::shape<CompiledShape::x_size>, nb::c_contig>;
    using MutablePackedArrayView   = nb::ndarray<nb::numpy, double, nb::shape<CompiledShape::x_size>, nb::c_contig>;
    using MutableJacobianArrayView = nb::ndarray<nb::numpy,
                                                double,
                                                nb::shape<CompiledShape::x_size, CompiledShape::x_size>,
                                                nb::c_contig>;
    using AlphaArrayView           = nb::ndarray<nb::numpy, const double, nb::shape<2>, nb::c_contig>;
    using RuntimeArrayView         = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;

    inline std::unique_ptr<SolveState>
    make_context(SolverKind solver, nonlinear::Workspace<CompiledShape::x_size>& workspace)
    {
        RuntimeCase input   = build_inline_case(0, 0, solver);
        auto      context = std::make_unique<SolveState>(input, workspace);

        context->raw_residual(std::span<const double, CompiledShape::x_size>{input.x0.data(), CompiledShape::x_size},
                              std::span<double, CompiledShape::x_size>{
                                  context->initial_raw.data(),
                                  CompiledShape::x_size,
                              });
        context->initial_alpha = {context->op.workspace.source_runtime.alpha1,
                                  context->op.workspace.source_runtime.alpha2};
        context->initial_raw_norm = norm2(std::span<const double, CompiledShape::x_size>{
            context->initial_raw.data(),
            CompiledShape::x_size,
        });
        context->input.residual_scale = build_residual_scale_for_context(*context, context->initial_raw);
        context->refresh_inverse_residual_scale();
        for (size_t i = 0; i < CompiledShape::x_size; ++i)
            context->initial_scaled[i] = context->initial_raw[i] / context->input.residual_scale[i];
        context->initial_scaled_norm = norm2(std::span<const double, CompiledShape::x_size>{
            context->initial_scaled.data(),
            CompiledShape::x_size,
        });
        context->has_initial_residual         = true;
        context->initial_residual_evaluations = 1 + residual_scale_extra_evaluations(context->input);
        return context;
    }

    template <size_t N>
    void read_runtime_prefix_array(RuntimeArrayView values,
                                   const char*      name,
                                   std::array<double, N>& out,
                                   size_t            required_length)
    {
        const size_t length = values.shape(0);
        if (length != required_length || length > N)
            throw std::runtime_error(std::string{name} + " length mismatch: expected " +
                                     std::to_string(required_length) + ", got " +
                                     std::to_string(length));
        out.fill(0.0);
        for (size_t i = 0; i < length; ++i)
            out[i] = values.data()[i];
    }

    template <size_t N>
    void read_runtime_offset_array(RuntimeArrayView       values,
                                   const char*            name,
                                   std::array<double, N>& out,
                                   bool                   sine_family)
    {
        const size_t length = values.shape(0);
        if (length > N)
            throw std::runtime_error(std::string{name} + " length mismatch: expected at most " + std::to_string(N) +
                                     ", got " + std::to_string(length));
        if (sine_family && length == N - 1)
        {
            for (size_t i = 0; i < length; ++i)
                out[i + 1] = values.data()[i];
            return;
        }
        for (size_t i = 0; i < length; ++i)
            out[i] = values.data()[i];
    }

    inline RuntimeCase runtime_case_from_abi(const std::string& case_name,
                                             double             a,
                                             double             R0,
                                             double             Z0,
                                             double             B0,
                                             double             ka,
                                             RuntimeArrayView   c_offsets,
                                             RuntimeArrayView   s_offsets,
                                             RuntimeArrayView   scaled_pprime,
                                             RuntimeArrayView   scaled_driver,
                                             RuntimeArrayView   source_nodes,
                                             RuntimeArrayView   pprime_coefficients,
                                             RuntimeArrayView   driver_coefficients,
                                             size_t             source_count,
                                             double             scaled_p0,
                                             double             scaled_Ip,
                                             double             beta,
                                             int                method_code,
                                             double             max_residual,
                                             int                max_evaluations,
                                             double             accepted_residual_factor,
                                             double             accepted_residual_floor,
                                             int                initial_policy_code,
                                             int                continue_policy_code,
                                             int                residual_normalization_code,
                                             double             residual_normalization_floor,
                                             double             residual_normalization_max_ratio,
                                             double             residual_normalization_huber_tau,
                                             int                residual_normalization_probe_count,
                                             double             residual_normalization_probe_step,
                                             double             residual_normalization_sensitivity_lambda)
    {
        RuntimeCase input = build_inline_case(0, 0, solver_kind_from_runtime_method_code(method_code));
        if (!case_name.empty())
            input.case_name = case_name;

        input.a  = a;
        input.R0 = R0;
        input.Z0 = Z0;
        input.B0 = B0;
        input.ka = ka;
        input.c_offsets.fill(0.0);
        input.s_offsets.fill(0.0);
        read_runtime_offset_array(c_offsets, "c_offsets", input.c_offsets, false);
        read_runtime_offset_array(s_offsets, "s_offsets", input.s_offsets, true);
        input.c0_offset = input.c_offsets[0];
        if constexpr (CompiledShape::M_max >= 1)
            input.s1_offset = input.s_offsets[1];

        if (source_count < 2 || source_count > CompiledSource::sample_count)
            throw std::runtime_error("source_count must be between 2 and " +
                                     std::to_string(CompiledSource::sample_count));
        read_runtime_prefix_array(scaled_pprime, "scaled_pprime", input.pprime, source_count);
        read_runtime_prefix_array(scaled_driver, "scaled_driver", input.driver, source_count);
        read_runtime_prefix_array(source_nodes, "source_nodes", input.source_nodes, source_count);
        read_runtime_prefix_array(
            pprime_coefficients,
            "pprime_coefficients",
            input.pprime_coefficients,
            4 * (source_count - 1));
        read_runtime_prefix_array(
            driver_coefficients,
            "driver_coefficients",
            input.driver_coefficients,
            4 * (source_count - 1));
        input.source_count = source_count;
        input.explicit_source_interpolation = true;

        input.p0      = scaled_p0;
        input.Ip      = scaled_Ip;
        input.beta    = beta;

        input.solver                   = solver_kind_from_runtime_method_code(method_code);
        input.max_residual             = max_residual;
        input.max_evaluations          = max_evaluations;
        if (input.max_evaluations < 0)
            throw std::runtime_error("max_evaluations must be non-negative");
        input.accepted_residual_factor = accepted_residual_factor;
        input.accepted_residual_floor  = accepted_residual_floor;
        input.initial_policy_code      = initial_policy_code;
        validate_initial_policy_code(input.initial_policy_code);
        input.continue_policy_code = continue_policy_code;
        validate_continue_policy_code(input.continue_policy_code);
        input.residual_normalization_code = residual_normalization_code;
        validate_residual_normalization_code(input.residual_normalization_code);
        input.residual_normalization_floor              = residual_normalization_floor;
        input.residual_normalization_max_ratio          = residual_normalization_max_ratio;
        input.residual_normalization_huber_tau          = residual_normalization_huber_tau;
        input.residual_normalization_probe_count        = residual_normalization_probe_count;
        input.residual_normalization_probe_step         = residual_normalization_probe_step;
        input.residual_normalization_sensitivity_lambda = residual_normalization_sensitivity_lambda;

        apply_initial_policy(input);

        input.x_scale = build_x_block_scale_vector<CompiledShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);
        return input;
    }

    inline double local_abs(double value) noexcept { return value < 0.0 ? -value : value; }

    inline bool cold_policy_uses_geometric_seed(const RuntimeCase& input, int policy_code) noexcept
    {
        if (policy_code == InitialPolicyColdGeometric || policy_code == ContinuePolicyColdGeometric)
            return true;
        if (policy_code == InitialPolicyCold || policy_code == ContinuePolicyCold)
            return boundary_curve_strain(input) >= 0.20;
        return false;
    }

    inline bool project_psin0_from_source_target(SolveState& context, double& coeff_out) noexcept
    {
        constexpr bool has_active_psin = CompiledShape::slot_for_profile_id(CompiledShape::psin_profile_id).optimized();
        if constexpr (!has_active_psin)
        {
            (void)context;
            (void)coeff_out;
            return false;
        }
        else
        {
            constexpr int psin0_index = CompiledShape::coeff_index[CompiledShape::psin_profile_id][0];
            if constexpr (psin0_index < 0)
            {
                (void)context;
                (void)coeff_out;
                return false;
            }
            else
            {
                PackedVector scratch{uninitialized};
                context.op.evaluate(
                    std::span<const double, CompiledShape::x_size>{
                        context.input.x0.data(),
                        CompiledShape::x_size,
                    },
                    scratch);

                constexpr double tiny                  = 1.0e-16;
                constexpr double coeff_damping         = 0.5;
                constexpr double coeff_abs_limit       = 1.0 - 1.0e-6;
                constexpr double radial_derivative_tol = 1.0e-10;
                constexpr double value_margin          = 5.0e-2;
                constexpr double monotonic_tol         = 1.0e-10;

                const double target_offset =
                    context.op.workspace.source_runtime.source_target_root_fields(source::root_psin, 0);
                const double target_scale = context.op.workspace.source_runtime.source_target_root_fields(
                                                source::root_psin, CompiledGrid::radial_nodes - 1) -
                                            target_offset;
                if (local_abs(target_scale) <= tiny)
                    return false;

                double denominator = 0.0;
                double numerator   = 0.0;
                double lower       = -coeff_abs_limit;
                double upper       = coeff_abs_limit;

                for (size_t i = 0; i < CompiledGrid::radial_nodes; ++i)
                {
                    double normalized =
                        (context.op.workspace.source_runtime.source_target_root_fields(source::root_psin, i) -
                         target_offset) /
                        target_scale;
                    if (i == 0)
                        normalized = 0.0;
                    else if (i + 1 == CompiledGrid::radial_nodes)
                        normalized = 1.0;

                    const double r     = CompiledGrid::nodes[i];
                    const double y       = CompiledGrid::y[i];
                    const double r2    = r * r;
                    const double base    = r2;
                    const double basis   = r2 * y;
                    const double weight  = CompiledGrid::weights[i];
                    const double base_r  = 2.0 * r;
                    const double basis_r = 2.0 * r * y - 2.0 * r * r2;
                    const double rhs     = radial_derivative_tol - base_r;

                    denominator += weight * basis * basis;
                    numerator += weight * basis * (normalized - base);

                    if (basis_r > tiny)
                    {
                        const double candidate = rhs / basis_r;
                        if (candidate > lower)
                            lower = candidate;
                    }
                    else if (basis_r < -tiny)
                    {
                        const double candidate = rhs / basis_r;
                        if (candidate < upper)
                            upper = candidate;
                    }
                    else if (base_r <= radial_derivative_tol)
                    {
                        return false;
                    }
                }

                if (denominator <= tiny || lower > upper)
                    return false;

                double coeff = coeff_damping * numerator / denominator;
                if (coeff < lower)
                    coeff = lower;
                else if (coeff > upper)
                    coeff = upper;
                if (2.0 * coeff * numerator - coeff * coeff * denominator <= 0.0)
                    return false;

                double previous = 0.0;
                for (size_t i = 0; i < CompiledGrid::radial_nodes; ++i)
                {
                    const double r               = CompiledGrid::nodes[i];
                    const double y                 = CompiledGrid::y[i];
                    const double r2              = r * r;
                    const double value             = r2 + coeff * r2 * y;
                    const double radial_derivative = 2.0 * r + coeff * (2.0 * r * y - 2.0 * r * r2);
                    if (radial_derivative <= radial_derivative_tol)
                        return false;
                    if (value < -value_margin || value > 1.0 + value_margin)
                        return false;
                    if (i > 0 && value - previous < -monotonic_tol)
                        return false;
                    previous = value;
                }

                coeff_out = coeff;
                return true;
            }
        }
    }

    inline void refine_cold_initial_state(SolveState& context, int policy_code)
    {
        if (!cold_policy_uses_geometric_seed(context.input, policy_code))
            return;
        double coeff = 0.0;
        if (!project_psin0_from_source_target(context, coeff))
            return;
        constexpr int psin0_index = CompiledShape::coeff_index[CompiledShape::psin_profile_id][0];
        if constexpr (psin0_index >= 0)
        {
            context.input.x0[static_cast<size_t>(psin0_index)] = coeff;
            context.input.x_scale =
                build_x_block_scale_vector<CompiledShape>(context.input.x0, profile_params_for_case(context.input));
        }
    }

    inline void refresh_initial_residual_scale(SolveState& context)
    {
        context.raw_residual(
            std::span<const double, CompiledShape::x_size>{
                context.input.x0.data(),
                CompiledShape::x_size,
            },
            std::span<double, CompiledShape::x_size>{context.initial_raw.data(), CompiledShape::x_size});
        context.initial_alpha = {context.op.workspace.source_runtime.alpha1,
                                 context.op.workspace.source_runtime.alpha2};
        context.initial_raw_norm = norm2(std::span<const double, CompiledShape::x_size>{
            context.initial_raw.data(),
            CompiledShape::x_size,
        });
        context.input.residual_scale = build_residual_scale_for_context(context, context.initial_raw);
        context.refresh_inverse_residual_scale();
        for (size_t i = 0; i < CompiledShape::x_size; ++i)
            context.initial_scaled[i] = context.initial_raw[i] / context.input.residual_scale[i];
        context.initial_scaled_norm = norm2(std::span<const double, CompiledShape::x_size>{
            context.initial_scaled.data(),
            CompiledShape::x_size,
        });
        context.has_initial_residual         = true;
        context.initial_residual_evaluations = 1 + residual_scale_extra_evaluations(context.input);
    }

    constexpr const char* source_route_name() noexcept
    {
        switch (Topology::source_route_code)
        {
        case Topology::SourceRoutePF:
            return "PF";
        case Topology::SourceRoutePP:
            return "PP";
        case Topology::SourceRoutePI:
            return "PI";
        case Topology::SourceRoutePJ1:
            return "PJ1";
        case Topology::SourceRoutePJ2:
            return "PJ2";
        case Topology::SourceRoutePQ:
            return "PQ";
        case Topology::SourceRoutePJ3:
            return "PJ3";
        default:
            return "unknown";
        }
    }

    constexpr const char* source_coordinate_name() noexcept
    {
        switch (Topology::source_coordinate_code)
        {
        case Topology::SourceCoordinateR:
            return "r";
        case Topology::SourceCoordinatePsin:
            return "psin";
        default:
            return "unknown";
        }
    }

    constexpr const char* source_constraint_name() noexcept
    {
        switch (Topology::source_constraint_code)
        {
        case Topology::SourceConstraintNull:
            return "null";
        case Topology::SourceConstraintIp:
            return "Ip";
        case Topology::SourceConstraintBeta:
            return "beta";
        case Topology::SourceConstraintIpBeta:
            return "Ip_beta";
        default:
            return "unknown";
        }
    }

    constexpr const char* source_nodes_name() noexcept
    {
        switch (Topology::source_nodes_code)
        {
        case Topology::SourceNodesUniform:
            return "uniform";
        case Topology::SourceNodesGrid:
            return "grid";
        default:
            return "unknown";
        }
    }

    inline std::string source_route_label()
    {
        return std::string{source_route_name()} + "/" + source_coordinate_name() + "/" + source_nodes_name() + "/" +
               source_constraint_name();
    }

    inline PackedArrayView packed_view(const double* data, nb::handle owner)
    {
        return PackedArrayView(data, {CompiledShape::x_size}, owner);
    }

    inline AlphaArrayView alpha_view(const double* data, nb::handle owner) { return AlphaArrayView(data, {2}, owner); }

    template <typename Counts>
    nb::list counts_list(const Counts& counts)
    {
        nb::list out;
        for (size_t value : counts)
            out.append(value);
        return out;
    }

    inline nb::dict topology_metadata_dict(const RuntimeCase& input)
    {
        nb::dict source;
        source["route"]        = source_route_name();
        source["route_code"]   = Topology::source_route_code;
        source["coordinate"]   = source_coordinate_name();
        source["coordinate_code"] = Topology::source_coordinate_code;
        source["constraint"]   = source_constraint_name();
        source["constraint_code"] = Topology::source_constraint_code;
        source["nodes"]        = source_nodes_name();
        source["nodes_code"]   = Topology::source_nodes_code;
        source["source_count"] = input.source_count;
        source["source_capacity"] = CompiledSource::sample_count;
        source["active_family_code"] = Topology::source_active_family_code;
        source["parameterization_code"] = Topology::source_parameterization_code;

        nb::dict grid;
        grid["Nr"]         = CompiledGrid::radial_nodes;
        grid["Nt"]         = CompiledGrid::theta_rows;
        grid["L_max"]      = CompiledShape::L_max;
        grid["M_max"]      = CompiledShape::M_max;
        grid["K_max"]      = CompiledShape::K_max;
        grid["quadrature"] = "legendre";
        grid["calculus"]   = "spectral";

        nb::dict profiles;
        using ProfileEvaluator  = profiles::ProfileEvaluator<CompiledShape>;
        profiles["h_count"]     = ProfileEvaluator::h_count;
        profiles["v_count"]     = ProfileEvaluator::v_count;
        profiles["kappa_count"] = ProfileEvaluator::kappa_count;
        profiles["psin_count"]  = ProfileEvaluator::psin_count;
        profiles["F_count"]     = ProfileEvaluator::F_count;
        profiles["c_counts"]    = counts_list(kernel_c_counts);
        profiles["s_counts"]    = counts_list(kernel_s_counts);

        nb::dict solver;
        solver["method_code"]                 = solver_method_code(input.solver);
        solver["method"]                      = solver_method(input.solver);
        solver["entrypoint"]                  = solver_entrypoint(input.solver);
        solver["jacobian"]                    = solver_jacobian(input);
#ifdef ENABLE_ENZYME
        if (input.solver != SolverKind::NewtonKrylov)
            solver["enzyme_jacobian_batch_width"] = enzyme_dense_jacobian_batch_width();
#endif
        solver["initial_policy_code"]         = input.initial_policy_code;
        solver["initial_policy"]              = initial_policy_name(input.initial_policy_code);
        solver["continue_policy_code"]        = input.continue_policy_code;
        solver["continue_policy"]             = continue_policy_name(input.continue_policy_code);
        solver["residual_normalization_code"] = input.residual_normalization_code;
        solver["residual_normalization"]      = residual_normalization_name(input.residual_normalization_code);

        nb::dict out;
        out["schema"]        = "veqpy.cxx.kernel.metadata.v2";
        out["backend"]       = "veqpy.cxx.nanobind";
        out["route"]         = source_route_label();
        out["x_size"]        = CompiledShape::x_size;
        out["active_count"]  = CompiledShape::active_count;
        out["source"]        = source;
        out["grid"]          = grid;
        out["profiles"]      = profiles;
        nb::dict layout;
        layout["profile_first"] = Topology::layout_profile_first;
        out["layout"]        = layout;
        out["solver"]        = solver;
        out["case_mutation"] = "typed_runtime";
        return out;
    }

    class NativeSolver
    {
    public:
        explicit NativeSolver(int solver_code = static_cast<int>(cxx_kernel_api::SolverMethodPowell))
            : solver_(solver_kind_from_runtime_method_code(solver_code)),
              context_(make_context(solver_, nonlinear_workspace_))
        {
        }

        nb::dict metadata() const { return topology_metadata_dict(context_->input); }

        nb::dict source_state() const
        {
            const auto& source_runtime = context_->op.workspace.source_runtime;
            nb::dict    out;
            out["alpha1"]              = source_runtime.alpha1;
            out["alpha2"]              = source_runtime.alpha2;
            out["scaled_effective_p0"] = source_runtime.scaled_effective_p0;
            out["pressure_multiplier"] = source_runtime.pressure_multiplier;
            return out;
        }

        // Typed runtime setter used by the Python Kernel hot path.  The Python layer
        // normalizes arrays to 1D float64/C-contiguous views; this C++ boundary
        // remains authoritative for topology-dependent lengths and enum validity.
        void set_kernel_runtime(const std::string& case_name,
                                double             a,
                                double             R0,
                                double             Z0,
                                double             B0,
                                double             ka,
                                RuntimeArrayView   c_offsets,
                                RuntimeArrayView   s_offsets,
                                RuntimeArrayView   scaled_pprime,
                                RuntimeArrayView   scaled_driver,
                                RuntimeArrayView   source_nodes,
                                RuntimeArrayView   pprime_coefficients,
                                RuntimeArrayView   driver_coefficients,
                                size_t             source_count,
                                double             scaled_p0,
                                double             scaled_Ip,
                                double             beta,
                                int                method_code,
                                double             max_residual,
                                int                max_evaluations,
                                double             accepted_residual_factor,
                                double             accepted_residual_floor,
                                int                initial_policy_code,
                                int                continue_policy_code,
                                int                residual_normalization_code,
                                double             residual_normalization_floor,
                                double             residual_normalization_max_ratio,
                                double             residual_normalization_huber_tau,
                                int                residual_normalization_probe_count,
                                double             residual_normalization_probe_step,
                                double             residual_normalization_sensitivity_lambda)
        {
            RuntimeCase next_input = runtime_case_from_abi(case_name,
                                                           a,
                                                           R0,
                                                           Z0,
                                                           B0,
                                                           ka,
                                                           c_offsets,
                                                           s_offsets,
                                                           scaled_pprime,
                                                           scaled_driver,
                                                           source_nodes,
                                                           pprime_coefficients,
                                                           driver_coefficients,
                                                           source_count,
                                                           scaled_p0,
                                                           scaled_Ip,
                                                           beta,
                                                           method_code,
                                                           max_residual,
                                                           max_evaluations,
                                                           accepted_residual_factor,
                                                           accepted_residual_floor,
                                                           initial_policy_code,
                                                           continue_policy_code,
                                                           residual_normalization_code,
                                                           residual_normalization_floor,
                                                           residual_normalization_max_ratio,
                                                           residual_normalization_huber_tau,
                                                           residual_normalization_probe_count,
                                                           residual_normalization_probe_step,
                                                           residual_normalization_sensitivity_lambda);
            apply_runtime_case(std::move(next_input));
        }

        void warmup(size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                SolveResult result = solve_current_case();
                if (accepted_result_succeeded(result))
                    record_accepted_result(result, false);
            }
        }

        void adopt_last_solution_as_initial()
        {
            if (!has_last_result_)
                throw std::runtime_error("NativeSolver has no solve result to adopt");
            if (!last_result_.accepted || !solver_info_succeeded(context_->input.solver, last_result_.info))
                throw std::runtime_error("NativeSolver cannot adopt an unsuccessful solve result");

            context_->input.x0      = last_result_.x;
            context_->input.x_scale = build_x_block_scale_vector<CompiledShape>(
                context_->input.x0,
                profile_params_for_case(context_->input));
            refresh_initial_residual_scale(*context_);
        }

        void set_initial_state(PackedArrayView x0)
        {
            std::copy_n(x0.data(), CompiledShape::x_size, context_->input.x0.begin());
            context_->input.x_scale = build_x_block_scale_vector<CompiledShape>(
                context_->input.x0,
                profile_params_for_case(context_->input));
            refresh_initial_residual_scale(*context_);
        }

        nb::tuple solve_direct()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_       = solve_current_case();
            has_last_result_   = true;
            if (accepted_result_succeeded(last_result_))
                record_accepted_result(last_result_, false);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_   = std::chrono::duration<double, std::milli>(elapsed).count();

            nb::object owner = nb::cast(this, nb::rv_policy::reference);
            return nb::make_tuple(last_elapsed_ms_,
                                  last_result_.accepted &&
                                      solver_info_succeeded(context_->input.solver, last_result_.info),
                                  last_result_.info,
                                  last_result_.nfev,
                                  last_result_.njev,
                                  last_result_.callbacks,
                                  last_result_.jacobian_component_evaluations,
                                  last_result_.jvp_evaluations,
                                  last_result_.linear_iterations,
                                  last_result_.raw_norm,
                                  last_result_.scaled_norm,
                                  packed_view(last_result_.x.data(), owner),
                                  packed_view(last_result_.raw.data(), owner),
                                  packed_view(last_result_.scaled.data(), owner),
                                  alpha_view(last_result_.alpha.data(), owner));
        }

        void residual_var_into(MutablePackedArrayView out, PackedArrayView x)
        {
            context_->raw_residual(std::span<const double, CompiledShape::x_size>{x.data(), CompiledShape::x_size},
                                   std::span<double, CompiledShape::x_size>{out.data(), CompiledShape::x_size});
        }

        void jvp_into(MutablePackedArrayView out, PackedArrayView x, PackedArrayView v)
        {
            constexpr size_t n = CompiledShape::x_size;
            double           v_norm_sq = 0.0;
            double           x_norm_sq = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                v_norm_sq += v.data()[i] * v.data()[i];
                x_norm_sq += x.data()[i] * x.data()[i];
            }
            const double v_norm = std::sqrt(v_norm_sq);
            if (v_norm <= 0.0)
            {
                std::fill(out.data(), out.data() + n, 0.0);
                return;
            }

            const double eps = std::sqrt(1.0e-12) * (1.0 + std::sqrt(x_norm_sq)) / v_norm;
            PackedVector x_plus{uninitialized};
            PackedVector f_base{uninitialized};
            PackedVector f_plus{uninitialized};
            for (size_t i = 0; i < n; ++i)
                x_plus[i] = x.data()[i] + eps * v.data()[i];

            context_->raw_residual(std::span<const double, n>{x.data(), n}, std::span<double, n>{f_base.data(), n});
            context_->raw_residual(std::span<const double, n>{x_plus.data(), n},
                                   std::span<double, n>{f_plus.data(), n});
            for (size_t i = 0; i < n; ++i)
                out.data()[i] = (f_plus[i] - f_base[i]) / eps;
        }

        void jacobian_into(MutableJacobianArrayView out, PackedArrayView x)
        {
            constexpr size_t n = CompiledShape::x_size;
            PackedVector     x_plus{uninitialized};
            PackedVector     f_base{uninitialized};
            PackedVector     f_plus{uninitialized};
            std::copy(x.data(), x.data() + n, x_plus.begin());
            context_->raw_residual(std::span<const double, n>{x.data(), n}, std::span<double, n>{f_base.data(), n});

            for (size_t col = 0; col < n; ++col)
            {
                const double saved = x_plus[col];
                const double step  = 1.0e-7 * std::max(1.0, std::abs(saved));
                x_plus[col]        = saved + step;
                context_->raw_residual(std::span<const double, n>{x_plus.data(), n},
                                       std::span<double, n>{f_plus.data(), n});
                x_plus[col] = saved;
                for (size_t row = 0; row < n; ++row)
                    out.data()[row * n + col] = (f_plus[row] - f_base[row]) / step;
            }
        }

        double last_elapsed_ms() const noexcept { return last_elapsed_ms_; }

    private:
        static constexpr double predictor_growth_limit = 10.0;
        static constexpr int    chord_max_iterations   = 3;

        bool accepted_result_succeeded(const SolveResult& result) const noexcept
        {
            return result.accepted && solver_info_succeeded(context_->input.solver, result.info);
        }

        bool same_array(const std::array<double, CompiledSource::sample_count>& lhs,
                        const std::array<double, CompiledSource::sample_count>& rhs) const noexcept
        {
            for (size_t i = 0; i < CompiledSource::sample_count; ++i)
                if (lhs[i] != rhs[i])
                    return false;
            return true;
        }

        template <size_t N>
        bool same_fixed_array(const std::array<double, N>& lhs,
                              const std::array<double, N>& rhs) const noexcept
        {
            for (size_t i = 0; i < N; ++i)
                if (lhs[i] != rhs[i])
                    return false;
            return true;
        }

        bool same_offsets(const std::array<double, CompiledShape::M_max + 1>& lhs,
                          const std::array<double, CompiledShape::M_max + 1>& rhs) const noexcept
        {
            for (size_t i = 0; i <= CompiledShape::M_max; ++i)
                if (lhs[i] != rhs[i])
                    return false;
            return true;
        }

        bool current_case_is_ip_only_reuse() const noexcept
        {
            if (!has_latest_input_)
                return false;
            const RuntimeCase& old = latest_input_;
            const RuntimeCase& now = context_->input;
            return old.a == now.a && old.R0 == now.R0 && old.Z0 == now.Z0 && old.B0 == now.B0 &&
                   old.ka == now.ka && old.c0_offset == now.c0_offset && old.s1_offset == now.s1_offset &&
                   old.p0 == now.p0 &&
                   same_offsets(old.c_offsets, now.c_offsets) && same_offsets(old.s_offsets, now.s_offsets) &&
                   old.explicit_source_interpolation == now.explicit_source_interpolation &&
                   same_array(old.pprime, now.pprime) && same_array(old.driver, now.driver) &&
                   same_array(old.source_nodes, now.source_nodes) &&
                   same_fixed_array(old.pprime_coefficients, now.pprime_coefficients) &&
                   same_fixed_array(old.driver_coefficients, now.driver_coefficients);
        }

        void fill_certified_result(SolveResult&                                   result,
                                   const std::array<double, CompiledShape::x_size>& x,
                                   const PackedVector&                            raw,
                                   const PackedVector&                            scaled,
                                   const std::array<double, 2>&                   alpha,
                                   double                                         raw_norm,
                                   double                                         scaled_norm,
                                   const std::string&                             accepted_by,
                                   const std::string&                             fast_path,
                                   int                                            certification_evals) const
        {
            result.x           = x;
            result.raw         = raw;
            result.scaled      = scaled;
            result.alpha       = alpha;
            result.raw_norm    = raw_norm;
            result.scaled_norm = scaled_norm;
            result.info        = 1;
            result.nfev        = context_->initial_residual_evaluations + certification_evals;
            result.solver_nfev = 0;
            result.njev        = 0;
            result.callbacks   = 0;
            result.jacobian_component_evaluations     = 0;
            result.jvp_evaluations                    = 0;
            result.linear_iterations                  = 0;
            result.initial_residual_evaluations       = context_->initial_residual_evaluations;
            result.certification_residual_evaluations = certification_evals;
            result.total_raw_residual_evaluations     = result.nfev;
            result.cert_threshold                     = acceptance_threshold(context_->input);
            result.initial_raw_norm                   = context_->initial_raw_norm;
            result.fast_path_raw_norm                 = raw_norm;
            result.accepted_by                        = accepted_by;
            result.fast_path                          = fast_path;
            result.fallback_used                      = false;
            result.fallback_reason                    = "";
            result.accepted                           = true;
        }

        bool try_initial_certificate(SolveResult& result) const
        {
            if (!has_latest_solution_ || !context_->has_initial_residual)
                return false;
            const double threshold = acceptance_threshold(context_->input);
            if (!math::is_finite(context_->initial_raw_norm) || context_->initial_raw_norm > threshold)
                return false;

            const std::string accepted_by =
                current_case_is_ip_only_reuse() ? "ip_only_warm_fixed" : "warm_fixed_certificate";
            fill_certified_result(result,
                                  context_->input.x0,
                                  context_->initial_raw,
                                  context_->initial_scaled,
                                  context_->initial_alpha,
                                  context_->initial_raw_norm,
                                  context_->initial_scaled_norm,
                                  accepted_by,
                                  "warm-fixed",
                                  0);
            return true;
        }

        void scale_raw_into(PackedVector& scaled, const PackedVector& raw) const noexcept
        {
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                scaled[i] = raw[i] / context_->input.residual_scale[i];
        }

        bool candidate_is_reasonable(const std::array<double, CompiledShape::x_size>& candidate) const noexcept
        {
            if (!has_latest_solution_)
                return false;
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
            {
                if (!math::is_finite(candidate[i]))
                    return false;
                const double delta = std::abs(candidate[i] - latest_solution_[i]);
                const double scale = std::max({std::abs(latest_solution_[i]), context_->input.x_scale[i], 1.0e-8});
                if (delta > predictor_growth_limit * scale)
                    return false;
            }
            return true;
        }

        bool try_candidate_certificate(const std::array<double, CompiledShape::x_size>& candidate,
                                       const std::string&                             accepted_by,
                                       const std::string&                             fast_path,
                                       int&                                           certification_evals,
                                       double&                                        best_raw_norm,
                                       SolveResult&                                   result)
        {
            if (!candidate_is_reasonable(candidate))
                return false;

            PackedVector raw{uninitialized};
            context_->raw_residual(
                std::span<const double, CompiledShape::x_size>{candidate.data(), CompiledShape::x_size},
                std::span<double, CompiledShape::x_size>{raw.data(), CompiledShape::x_size});
            const std::array<double, 2> alpha = {context_->op.workspace.source_runtime.alpha1,
                                                 context_->op.workspace.source_runtime.alpha2};
            ++certification_evals;

            const double raw_norm = norm2(std::span<const double, CompiledShape::x_size>{raw.data(), CompiledShape::x_size});
            if (math::is_finite(raw_norm) && raw_norm < best_raw_norm)
                best_raw_norm = raw_norm;
            if (!math::is_finite(raw_norm) || raw_norm > acceptance_threshold(context_->input))
                return false;

            PackedVector scaled{uninitialized};
            scale_raw_into(scaled, raw);
            const double scaled_norm =
                norm2(std::span<const double, CompiledShape::x_size>{scaled.data(), CompiledShape::x_size});
            fill_certified_result(result,
                                  candidate,
                                  raw,
                                  scaled,
                                  alpha,
                                  raw_norm,
                                  scaled_norm,
                                  accepted_by,
                                  fast_path,
                                  certification_evals);
            return true;
        }

        bool has_predictor_history() const noexcept { return has_latest_solution_ && has_previous_solution_; }

        bool try_predictor_certificates(int& certification_evals, double& best_raw_norm, SolveResult& result)
        {
            if (!has_predictor_history())
                return false;

            std::array<double, CompiledShape::x_size> secant{};
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                secant[i] = latest_solution_[i] + (latest_solution_[i] - previous_solution_[i]);
            if (try_candidate_certificate(
                    secant, "secant_predictor_certificate", "warm-predict", certification_evals, best_raw_norm, result))
                return true;

            if (!has_older_solution_)
                return false;

            std::array<double, CompiledShape::x_size> quadratic{};
            for (size_t i = 0; i < CompiledShape::x_size; ++i)
                quadratic[i] = 3.0 * latest_solution_[i] - 3.0 * previous_solution_[i] + older_solution_[i];
            return try_candidate_certificate(quadratic,
                                             "quadratic_predictor_certificate",
                                             "warm-predict",
                                             certification_evals,
                                             best_raw_norm,
                                             result);
        }

        void build_chord_jacobian(const std::array<double, CompiledShape::x_size>& current,
                                  const PackedVector&                            current_raw,
                                  tensor::Matrix<double, CompiledShape::x_size, CompiledShape::x_size>& jacobian,
                                  int&                                           certification_evals)
        {
            for (size_t col = 0; col < CompiledShape::x_size; ++col)
            {
                std::array<double, CompiledShape::x_size> perturbed = current;
                const double step = 1.0e-6 * std::max(1.0, std::abs(current[col]));
                perturbed[col] += step;

                PackedVector plus_raw{uninitialized};
                context_->raw_residual(
                    std::span<const double, CompiledShape::x_size>{perturbed.data(), CompiledShape::x_size},
                    std::span<double, CompiledShape::x_size>{plus_raw.data(), CompiledShape::x_size});
                ++certification_evals;

                for (size_t row = 0; row < CompiledShape::x_size; ++row)
                    jacobian[row * CompiledShape::x_size + col] = (plus_raw[row] - current_raw[row]) / step;
            }
        }

        bool try_chord_newton_certificate(int& certification_evals, double& best_raw_norm, SolveResult& result)
        {
            std::array<double, CompiledShape::x_size> current = context_->input.x0;
            PackedVector                            current_raw = context_->initial_raw;
            tensor::Matrix<double, CompiledShape::x_size, CompiledShape::x_size> chord_jacobian{uninitialized};
            build_chord_jacobian(current, current_raw, chord_jacobian, certification_evals);

            for (int iteration = 0; iteration < chord_max_iterations; ++iteration)
            {
                tensor::Matrix<double, CompiledShape::x_size, 1> rhs{uninitialized};
                for (size_t row = 0; row < CompiledShape::x_size; ++row)
                    rhs[row] = -current_raw[row];

                tensor::Matrix<double, CompiledShape::x_size, 1> step{uninitialized};
                linalg::solve_into(step, chord_jacobian, rhs);

                bool finite_step = true;
                for (size_t i = 0; i < CompiledShape::x_size; ++i)
                    finite_step = finite_step && math::is_finite(step[i]);
                if (!finite_step)
                    return false;

                std::array<double, CompiledShape::x_size> trial{};
                for (size_t i = 0; i < CompiledShape::x_size; ++i)
                    trial[i] = current[i] + step[i];

                PackedVector trial_raw{uninitialized};
                context_->raw_residual(
                    std::span<const double, CompiledShape::x_size>{trial.data(), CompiledShape::x_size},
                    std::span<double, CompiledShape::x_size>{trial_raw.data(), CompiledShape::x_size});
                const std::array<double, 2> trial_alpha = {context_->op.workspace.source_runtime.alpha1,
                                                           context_->op.workspace.source_runtime.alpha2};
                ++certification_evals;
                const double trial_norm =
                    norm2(std::span<const double, CompiledShape::x_size>{trial_raw.data(), CompiledShape::x_size});
                if (!math::is_finite(trial_norm))
                    return false;

                if (trial_norm <= acceptance_threshold(context_->input))
                {
                    PackedVector scaled{uninitialized};
                    scale_raw_into(scaled, trial_raw);
                    const double scaled_norm =
                        norm2(std::span<const double, CompiledShape::x_size>{scaled.data(), CompiledShape::x_size});
                    fill_certified_result(result,
                                          trial,
                                          trial_raw,
                                          scaled,
                                          trial_alpha,
                                          trial_norm,
                                          scaled_norm,
                                          "chord_newton_certificate",
                                          "warm-chord",
                                          certification_evals);
                    return true;
                }

                if (trial_norm >= best_raw_norm)
                    return false;
                best_raw_norm = trial_norm;
                current       = trial;
                current_raw   = trial_raw;
            }
            return false;
        }

        SolveResult solve_current_case()
        {
            const int policy = resolved_continue_policy(context_->input.continue_policy_code);
            if (!has_latest_solution_ || continue_policy_is_cold(policy))
                return run_solver_once(*context_);

            SolveResult fast_result{};
            if (try_initial_certificate(fast_result))
                return fast_result;

            int    certification_evals = 0;
            double best_raw_norm       = context_->has_initial_residual
                                           ? context_->initial_raw_norm
                                           : std::numeric_limits<double>::max();

            if (continue_policy_uses_predictor(policy) &&
                try_predictor_certificates(certification_evals, best_raw_norm, fast_result))
                return fast_result;

            if (continue_policy_uses_chord(policy) &&
                try_chord_newton_certificate(certification_evals, best_raw_norm, fast_result))
                return fast_result;

            SolveResult fallback = run_solver_once(*context_);
            fallback.nfev += certification_evals;
            fallback.certification_residual_evaluations = certification_evals;
            fallback.total_raw_residual_evaluations += certification_evals;
            fallback.accepted_by = std::string{solver_method(context_->input.solver)} + "_fallback";
            fallback.fast_path   = continue_policy_name(context_->input.continue_policy_code);
            fallback.fallback_used = true;
            fallback.fallback_reason =
                math::is_finite(best_raw_norm) ? "fast_path_raw_norm_above_threshold" : "fast_path_nonfinite_raw_norm";
            fallback.fast_path_raw_norm = best_raw_norm;
            return fallback;
        }

        void record_accepted_result(const SolveResult& result, bool refresh_scale)
        {
            context_->input.x0 = result.x;
            context_->input.x_scale =
                build_x_block_scale_vector<CompiledShape>(context_->input.x0, profile_params_for_case(context_->input));
            if (refresh_scale)
                refresh_initial_residual_scale(*context_);

            if (has_previous_solution_)
            {
                older_solution_     = previous_solution_;
                has_older_solution_ = true;
            }
            if (has_latest_solution_)
            {
                previous_solution_     = latest_solution_;
                has_previous_solution_ = true;
            }
            latest_solution_     = result.x;
            has_latest_solution_ = true;
            latest_input_        = context_->input;
            has_latest_input_    = true;
        }

        void apply_runtime_case(RuntimeCase next_input)
        {
            bool should_refine_cold = true;
            int  cold_policy_code   = next_input.initial_policy_code;
            const int continue_policy = resolved_continue_policy(next_input.continue_policy_code);
            if (has_latest_solution_)
            {
                if (continue_policy_uses_warm_state(continue_policy))
                {
                    next_input.x0      = latest_solution_;
                    next_input.x_scale = build_x_block_scale_vector<CompiledShape>(
                        next_input.x0,
                        profile_params_for_case(next_input));
                    should_refine_cold = false;
                }
                else if (continue_policy_is_cold(continue_policy))
                {
                    apply_cold_policy(next_input, continue_policy);
                    next_input.x_scale = build_x_block_scale_vector<CompiledShape>(
                        next_input.x0,
                        profile_params_for_case(next_input));
                    cold_policy_code = continue_policy;
                }
            }
            context_->reset_case(next_input);
            if (should_refine_cold)
                refine_cold_initial_state(*context_, cold_policy_code);
            refresh_initial_residual_scale(*context_);
        }

        SolverKind                    solver_;
        nonlinear::Workspace<CompiledShape::x_size> nonlinear_workspace_{};
        std::unique_ptr<SolveState> context_;
        SolveResult                   last_result_{};
        std::array<double, CompiledShape::x_size> older_solution_{};
        std::array<double, CompiledShape::x_size> previous_solution_{};
        std::array<double, CompiledShape::x_size> latest_solution_{};
        RuntimeCase                     latest_input_{};
        bool                          has_last_result_ = false;
        bool                          has_older_solution_ = false;
        bool                          has_previous_solution_ = false;
        bool                          has_latest_solution_ = false;
        bool                          has_latest_input_ = false;
        double                        last_elapsed_ms_ = 0.0;
    };

} // namespace cxx_python
