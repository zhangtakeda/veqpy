#pragma once

// Nanobind-facing KernelSolver implementation for VEQlib production kernels.

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nlohmann/json.hpp>

#include "kernel_case.h"
#include "kernel_runtime.h"
#include "kernel_topology.h"
#include "tensor.h"

namespace nb = nanobind;

namespace veqlib_python
{
    using std::size_t;

    using config::Topology;
    using veqlib_kernel_api::KernelGrid;
    using veqlib_kernel_api::KernelShape;
    using veqlib_kernel_api::KernelSource;
    using veqlib_kernel_api::CaseInput;
    using veqlib_kernel_api::InitialPolicyColdGeometric;
    using veqlib_kernel_api::InitialPolicyCold;
    using veqlib_kernel_api::InitialPolicyWarmClone;
    using veqlib_kernel_api::PackedVector;
    using veqlib_kernel_api::SolveContext;
    using veqlib_kernel_api::SolveResult;
    using veqlib_kernel_api::SolverKind;
    using veqlib_kernel_api::build_residual_scale_for_context;
    using veqlib_kernel_api::build_x_block_scale_vector;
    using veqlib_kernel_api::build_inline_case;
    using veqlib_kernel_api::apply_initial_policy;
    using veqlib_kernel_api::boundary_curve_strain;
    using veqlib_kernel_api::json_array;
    using veqlib_kernel_api::norm2;
    using veqlib_kernel_api::profile_params_for_case;
    using veqlib_kernel_api::run_solver_once;
    using veqlib_kernel_api::solve_result_json;
    using veqlib_kernel_api::solver_entrypoint;
    using veqlib_kernel_api::solver_info_succeeded;
    using veqlib_kernel_api::solver_jacobian;
    using veqlib_kernel_api::solver_kind_from_runtime_method_code;
    using veqlib_kernel_api::solver_method_code;
    using veqlib_kernel_api::solver_method;
    using veqlib_kernel_api::initial_policy_is_warm_clone;
    using veqlib_kernel_api::initial_policy_name;
    using veqlib_kernel_api::residual_normalization_name;
    using veqlib_kernel_api::validate_initial_policy_code;
    using veqlib_kernel_api::validate_residual_normalization_code;
    using veqlib_kernel_api::kernel_c_counts;
    using veqlib_kernel_api::kernel_s_counts;
#ifdef ENABLE_ENZYME
    using veqlib_kernel_api::enzyme_dense_jacobian_batch_width;
#endif

    using tensor::uninitialized;

    using PackedArrayView        = nb::ndarray<nb::numpy, const double, nb::shape<KernelShape::x_size>, nb::c_contig>;
    using MutablePackedArrayView = nb::ndarray<nb::numpy, double, nb::shape<KernelShape::x_size>, nb::c_contig>;
    using AlphaArrayView         = nb::ndarray<nb::numpy, const double, nb::shape<2>, nb::c_contig>;
    using RuntimeArrayView       = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;

    inline std::unique_ptr<SolveContext> make_context(SolverKind solver)
    {
        CaseInput input   = build_inline_case(0, 0, solver);
        auto      context = std::make_unique<SolveContext>(input);

        PackedVector initial_raw{uninitialized};
        context->raw_residual(std::span<const double, KernelShape::x_size>{input.x0.data(), KernelShape::x_size},
                              std::span<double, KernelShape::x_size>{initial_raw.data(), KernelShape::x_size});
        context->input.residual_scale = build_residual_scale_for_context(*context, initial_raw);
        return context;
    }

    inline const nlohmann::json* find_object(const nlohmann::json& data, const char* name)
    {
        const auto it = data.find(name);
        if (it == data.end() || it->is_null())
            return nullptr;
        if (!it->is_object())
            throw std::runtime_error(std::string{name} + " must be a JSON object");
        return &*it;
    }

    inline const nlohmann::json* find_array_field(const nlohmann::json& data, const char* name)
    {
        const auto it = data.find(name);
        if (it == data.end() || it->is_null())
            return nullptr;
        if (!it->is_array())
            throw std::runtime_error(std::string{name} + " must be a JSON array");
        return &*it;
    }

    inline bool has_array_field(const nlohmann::json& data, const char* name)
    {
        return find_array_field(data, name) != nullptr;
    }

    inline double finite_number(const nlohmann::json& data, const char* name)
    {
        const auto it = data.find(name);
        if (it == data.end() || !it->is_number())
            throw std::runtime_error(std::string{name} + " must be a JSON number");
        return it->get<double>();
    }

    inline double optional_finite_number(const nlohmann::json& data, const char* name, double fallback)
    {
        const auto it = data.find(name);
        if (it == data.end() || it->is_null())
            return fallback;
        if (!it->is_number())
            throw std::runtime_error(std::string{name} + " must be a JSON number");
        return it->get<double>();
    }

    inline int required_int(const nlohmann::json& data, const char* name)
    {
        const auto it = data.find(name);
        if (it == data.end() || !it->is_number_integer())
            throw std::runtime_error(std::string{name} + " must be a JSON integer");
        return it->get<int>();
    }

    inline int optional_int(const nlohmann::json& data, const char* name, int fallback)
    {
        const auto it = data.find(name);
        if (it == data.end() || it->is_null())
            return fallback;
        if (!it->is_number_integer())
            throw std::runtime_error(std::string{name} + " must be a JSON integer");
        return it->get<int>();
    }

    template <size_t N>
    void read_exact_array(const nlohmann::json& data, const char* name, std::array<double, N>& out)
    {
        const auto* values = find_array_field(data, name);
        if (values == nullptr)
            throw std::runtime_error(std::string{name} + " is required");
        if (values->size() != N)
            throw std::runtime_error(std::string{name} + " length mismatch: expected " + std::to_string(N) + ", got " +
                                     std::to_string(values->size()));
        for (size_t i = 0; i < N; ++i)
        {
            if (!(*values)[i].is_number())
                throw std::runtime_error(std::string{name} + " entries must be JSON numbers");
            out[i] = (*values)[i].get<double>();
        }
    }

    template <size_t N>
    void read_optional_offset_array(const nlohmann::json&  data,
                                    const char*            name,
                                    std::array<double, N>& out,
                                    bool                   sine_family)
    {
        const auto* values = find_array_field(data, name);
        if (values == nullptr)
            return;
        if (values->size() > N)
            throw std::runtime_error(std::string{name} + " length mismatch: expected at most " + std::to_string(N) +
                                     ", got " + std::to_string(values->size()));
        if (sine_family && values->size() == N - 1)
        {
            for (size_t i = 0; i < values->size(); ++i)
            {
                if (!(*values)[i].is_number())
                    throw std::runtime_error(std::string{name} + " entries must be JSON numbers");
                out[i + 1] = (*values)[i].get<double>();
            }
            return;
        }
        for (size_t i = 0; i < values->size(); ++i)
        {
            if (!(*values)[i].is_number())
                throw std::runtime_error(std::string{name} + " entries must be JSON numbers");
            out[i] = (*values)[i].get<double>();
        }
    }

    inline const nlohmann::json& required_object(const nlohmann::json& data, const char* name)
    {
        const auto* object = find_object(data, name);
        if (object == nullptr)
            throw std::runtime_error(std::string{name} + " is required");
        return *object;
    }

    inline const nlohmann::json& object_or_root(const nlohmann::json& data, const char* name)
    {
        const auto* object = find_object(data, name);
        return object == nullptr ? data : *object;
    }

    inline void read_exact_runtime_array(RuntimeArrayView values,
                                         const char*      name,
                                         std::array<double, KernelSource::sample_count>& out)
    {
        const size_t length = values.shape(0);
        if (length != KernelSource::sample_count)
            throw std::runtime_error(std::string{name} + " length mismatch: expected " +
                                     std::to_string(KernelSource::sample_count) + ", got " +
                                     std::to_string(length));
        for (size_t i = 0; i < KernelSource::sample_count; ++i)
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

    inline CaseInput case_input_from_json(const nlohmann::json& data, SolverKind solver)
    {
        CaseInput input = build_inline_case(0, 0, solver);
        if (const auto it = data.find("case_name"); it != data.end() && !it->is_null())
        {
            if (!it->is_string())
                throw std::runtime_error("case_name must be a string");
            input.case_name = it->get<std::string>();
        }

        const nlohmann::json& solver_config = required_object(data, "solver");
        input.solver          = solver_kind_from_runtime_method_code(required_int(solver_config, "method_code"));
        input.max_residual    = finite_number(solver_config, "max_residual");
        input.max_evaluations = required_int(solver_config, "max_evaluations");
        if (input.max_evaluations < 0)
            throw std::runtime_error("solver.max_evaluations must be non-negative");
        input.accepted_residual_factor = finite_number(solver_config, "accepted_residual_factor");
        input.accepted_residual_floor  = finite_number(solver_config, "accepted_residual_floor");
        input.initial_policy_code      = required_int(solver_config, "initial_policy_code");
        validate_initial_policy_code(input.initial_policy_code);
        input.residual_normalization_code = required_int(solver_config, "residual_normalization_code");
        validate_residual_normalization_code(input.residual_normalization_code);
        input.residual_normalization_floor       = finite_number(solver_config, "residual_normalization_floor");
        input.residual_normalization_max_ratio   = finite_number(solver_config, "residual_normalization_max_ratio");
        input.residual_normalization_huber_tau   = finite_number(solver_config, "residual_normalization_huber_tau");
        input.residual_normalization_probe_count = required_int(solver_config, "residual_normalization_probe_count");
        input.residual_normalization_probe_step  = finite_number(solver_config, "residual_normalization_probe_step");
        input.residual_normalization_sensitivity_lambda =
            finite_number(solver_config, "residual_normalization_sensitivity_lambda");

        const nlohmann::json& boundary = required_object(data, "boundary");
        input.a                        = finite_number(boundary, "a");
        input.R0                       = finite_number(boundary, "R0");
        input.Z0                       = optional_finite_number(boundary, "Z0", 0.0);
        input.B0                       = finite_number(boundary, "B0");
        input.ka                       = finite_number(boundary, "ka");
        input.c0_offset                = optional_finite_number(boundary, "c0_offset", input.c0_offset);
        input.s1_offset                = optional_finite_number(boundary, "s1_offset", input.s1_offset);
        input.c_offsets.fill(0.0);
        input.s_offsets.fill(0.0);
        input.c_offsets[0] = input.c0_offset;
        if constexpr (KernelShape::M_max >= 1)
            input.s_offsets[1] = input.s1_offset;
        read_optional_offset_array(boundary, "c_offsets", input.c_offsets, false);
        read_optional_offset_array(boundary, "s_offsets", input.s_offsets, true);
        input.c0_offset = input.c_offsets[0];
        if constexpr (KernelShape::M_max >= 1)
            input.s1_offset = input.s_offsets[1];

        const nlohmann::json& source = object_or_root(data, "source");
        read_exact_array(source, "scaled_heat", input.heat);
        read_exact_array(source, "scaled_current", input.current);

        const nlohmann::json& constraints = object_or_root(data, "constraints");
        input.Ip =
            optional_finite_number(constraints, "scaled_Ip", optional_finite_number(data, "scaled_Ip", input.Ip));
        input.beta = optional_finite_number(constraints, "beta", optional_finite_number(data, "beta", input.beta));
        input.fix_rho =
            optional_finite_number(constraints, "fix_rho", optional_finite_number(data, "fix_rho", input.fix_rho));

        if (input.initial_policy_code == InitialPolicyWarmClone)
            input.x0.fill(0.0);
        else
            apply_initial_policy(input);

        input.x_scale = build_x_block_scale_vector<KernelShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);

        return input;
    }

    inline CaseInput case_input_from_runtime(const std::string& case_name,
                                             double             a,
                                             double             R0,
                                             double             Z0,
                                             double             B0,
                                             double             ka,
                                             RuntimeArrayView   c_offsets,
                                             RuntimeArrayView   s_offsets,
                                             RuntimeArrayView   scaled_heat,
                                             RuntimeArrayView   scaled_current,
                                             double             scaled_Ip,
                                             double             beta,
                                             double             fix_rho,
                                             int                method_code,
                                             double             max_residual,
                                             int                max_evaluations,
                                             double             accepted_residual_factor,
                                             double             accepted_residual_floor,
                                             int                initial_policy_code,
                                             int                residual_normalization_code,
                                             double             residual_normalization_floor,
                                             double             residual_normalization_max_ratio,
                                             double             residual_normalization_huber_tau,
                                             int                residual_normalization_probe_count,
                                             double             residual_normalization_probe_step,
                                             double             residual_normalization_sensitivity_lambda)
    {
        CaseInput input = build_inline_case(0, 0, solver_kind_from_runtime_method_code(method_code));
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
        if constexpr (KernelShape::M_max >= 1)
            input.s1_offset = input.s_offsets[1];

        read_exact_runtime_array(scaled_heat, "scaled_heat", input.heat);
        read_exact_runtime_array(scaled_current, "scaled_current", input.current);

        input.Ip      = scaled_Ip;
        input.beta    = beta;
        input.fix_rho = fix_rho;

        input.solver                   = solver_kind_from_runtime_method_code(method_code);
        input.max_residual             = max_residual;
        input.max_evaluations          = max_evaluations;
        if (input.max_evaluations < 0)
            throw std::runtime_error("max_evaluations must be non-negative");
        input.accepted_residual_factor = accepted_residual_factor;
        input.accepted_residual_floor  = accepted_residual_floor;
        input.initial_policy_code      = initial_policy_code;
        validate_initial_policy_code(input.initial_policy_code);
        input.residual_normalization_code = residual_normalization_code;
        validate_residual_normalization_code(input.residual_normalization_code);
        input.residual_normalization_floor              = residual_normalization_floor;
        input.residual_normalization_max_ratio          = residual_normalization_max_ratio;
        input.residual_normalization_huber_tau          = residual_normalization_huber_tau;
        input.residual_normalization_probe_count        = residual_normalization_probe_count;
        input.residual_normalization_probe_step         = residual_normalization_probe_step;
        input.residual_normalization_sensitivity_lambda = residual_normalization_sensitivity_lambda;

        if (input.initial_policy_code == InitialPolicyWarmClone)
            input.x0.fill(0.0);
        else
            apply_initial_policy(input);

        input.x_scale = build_x_block_scale_vector<KernelShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);
        return input;
    }

    inline double local_abs(double value) noexcept { return value < 0.0 ? -value : value; }

    inline bool cold_policy_uses_geometric_seed(const CaseInput& input) noexcept
    {
        if (input.initial_policy_code == InitialPolicyColdGeometric)
            return true;
        if (input.initial_policy_code == InitialPolicyCold)
            return boundary_curve_strain(input) >= 0.20;
        return false;
    }

    inline bool project_psin0_from_source_target(SolveContext& context, double& coeff_out) noexcept
    {
        constexpr bool has_active_psin = KernelShape::slot_for_profile_id(KernelShape::psin_profile_id).optimized();
        if constexpr (!has_active_psin)
        {
            (void)context;
            (void)coeff_out;
            return false;
        }
        else
        {
            constexpr int psin0_index = KernelShape::coeff_index[KernelShape::psin_profile_id][0];
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
                    std::span<const double, KernelShape::x_size>{
                        context.input.x0.data(),
                        KernelShape::x_size,
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
                                                source::root_psin, KernelGrid::radial_nodes - 1) -
                                            target_offset;
                if (local_abs(target_scale) <= tiny)
                    return false;

                double denominator = 0.0;
                double numerator   = 0.0;
                double lower       = -coeff_abs_limit;
                double upper       = coeff_abs_limit;

                for (size_t i = 0; i < KernelGrid::radial_nodes; ++i)
                {
                    double normalized =
                        (context.op.workspace.source_runtime.source_target_root_fields(source::root_psin, i) -
                         target_offset) /
                        target_scale;
                    if (i == 0)
                        normalized = 0.0;
                    else if (i + 1 == KernelGrid::radial_nodes)
                        normalized = 1.0;

                    const double rho     = KernelGrid::nodes[i];
                    const double y       = KernelGrid::y[i];
                    const double rho2    = rho * rho;
                    const double base    = rho2;
                    const double basis   = rho2 * y;
                    const double weight  = KernelGrid::weights[i];
                    const double base_r  = 2.0 * rho;
                    const double basis_r = 2.0 * rho * y - 2.0 * rho * rho2;
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
                for (size_t i = 0; i < KernelGrid::radial_nodes; ++i)
                {
                    const double rho               = KernelGrid::nodes[i];
                    const double y                 = KernelGrid::y[i];
                    const double rho2              = rho * rho;
                    const double value             = rho2 + coeff * rho2 * y;
                    const double radial_derivative = 2.0 * rho + coeff * (2.0 * rho * y - 2.0 * rho * rho2);
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

    inline void refine_cold_initial_state(SolveContext& context)
    {
        if (!cold_policy_uses_geometric_seed(context.input))
            return;
        double coeff = 0.0;
        if (!project_psin0_from_source_target(context, coeff))
            return;
        constexpr int psin0_index = KernelShape::coeff_index[KernelShape::psin_profile_id][0];
        if constexpr (psin0_index >= 0)
        {
            context.input.x0[static_cast<size_t>(psin0_index)] = coeff;
            context.input.x_scale =
                build_x_block_scale_vector<KernelShape>(context.input.x0, profile_params_for_case(context.input));
        }
    }

    inline void refresh_initial_residual_scale(SolveContext& context)
    {
        PackedVector initial_raw{uninitialized};
        context.raw_residual(
            std::span<const double, KernelShape::x_size>{
                context.input.x0.data(),
                KernelShape::x_size,
            },
            std::span<double, KernelShape::x_size>{initial_raw.data(), KernelShape::x_size});
        context.input.residual_scale = build_residual_scale_for_context(context, initial_raw);
    }

    inline nlohmann::json solver_json(const CaseInput& input)
    {
        return {
            {"method_code", solver_method_code(input.solver)},
            {"max_residual", input.max_residual},
            {"max_evaluations", input.max_evaluations},
            {"accepted_residual_factor", input.accepted_residual_factor},
            {"accepted_residual_floor", input.accepted_residual_floor},
            {"initial_policy_code", input.initial_policy_code},
            {"residual_normalization_code", input.residual_normalization_code},
            {"residual_normalization_floor", input.residual_normalization_floor},
            {"residual_normalization_max_ratio", input.residual_normalization_max_ratio},
            {"residual_normalization_huber_tau", input.residual_normalization_huber_tau},
            {"residual_normalization_probe_count", input.residual_normalization_probe_count},
            {"residual_normalization_probe_step", input.residual_normalization_probe_step},
            {"residual_normalization_sensitivity_lambda", input.residual_normalization_sensitivity_lambda},
        };
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
        default:
            return "unknown";
        }
    }

    constexpr const char* source_coordinate_name() noexcept
    {
        switch (Topology::source_coordinate_code)
        {
        case Topology::SourceCoordinateRho:
            return "rho";
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

    inline nlohmann::json case_prefix_json(SolveContext& context)
    {
        const CaseInput& input = context.input;

        return {
            {"case_name", input.case_name},
            {"route", source_route_label()},
            {"source_topology",
             {
                 {"route", source_route_name()},
                 {"coordinate", source_coordinate_name()},
                 {"nodes", source_nodes_name()},
                 {"constraint", source_constraint_name()},
             }},
            {"x_size", KernelShape::x_size},
            {"grid",
             {
                 {"Nr", KernelGrid::radial_nodes},
                 {"Nt", KernelGrid::theta_rows},
                 {"L_max", KernelShape::L_max},
                 {"M_max", KernelShape::M_max},
                 {"K_max", KernelShape::K_max},
                 {"quadrature_scheme", "legendre"},
                 {"calculus_scheme", "spectral"},
             }},
            {"boundary",
             {
                 {"a", input.a},
                 {"R0", input.R0},
                 {"Z0", input.Z0},
                 {"B0", input.B0},
                 {"ka", input.ka},
                 {"c0_offset", input.c0_offset},
                 {"s1_offset", input.s1_offset},
                 {"c_offsets", json_array(input.c_offsets)},
                 {"s_offsets", json_array(input.s_offsets)},
             }},
            {"solver", solver_json(input)},
            {"source",
             {
                 {"scaled_heat", json_array(input.heat)},
                 {"scaled_current", json_array(input.current)},
             }},
            {"constraints", {{"scaled_Ip", input.Ip}, {"beta", input.beta}, {"fix_rho", input.fix_rho}}},
            {"fix_rho", input.fix_rho},
        };
    }

    inline PackedArrayView packed_view(const double* data, nb::handle owner)
    {
        return PackedArrayView(data, {KernelShape::x_size}, owner);
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

    inline nb::dict topology_metadata_dict(const CaseInput& input)
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
        source["sample_count"] = KernelSource::sample_count;
        source["active_family_code"] = Topology::source_active_family_code;
        source["parameterization_code"] = Topology::source_parameterization_code;

        nb::dict grid;
        grid["Nr"]         = KernelGrid::radial_nodes;
        grid["Nt"]         = KernelGrid::theta_rows;
        grid["L_max"]      = KernelShape::L_max;
        grid["M_max"]      = KernelShape::M_max;
        grid["K_max"]      = KernelShape::K_max;
        grid["quadrature"] = "legendre";
        grid["calculus"]   = "spectral";

        nb::dict profiles;
        using ProfileEvaluator  = profiles::ProfileEvaluator<KernelShape>;
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
        solver["residual_normalization_code"] = input.residual_normalization_code;
        solver["residual_normalization"]      = residual_normalization_name(input.residual_normalization_code);

        nb::dict out;
        out["schema"]        = "veqlib.kernel.metadata.v1";
        out["backend"]       = "veqlib.nanobind";
        out["route"]         = source_route_label();
        out["x_size"]        = KernelShape::x_size;
        out["active_count"]  = KernelShape::active_count;
        out["source"]        = source;
        out["grid"]          = grid;
        out["profiles"]      = profiles;
        nb::dict layout;
        layout["profile_first"] = Topology::layout_profile_first;
        out["layout"]        = layout;
        out["solver"]        = solver;
        out["case_mutation"] = "json_payload_from_topology";
        return out;
    }

    class KernelSolver
    {
    public:
        explicit KernelSolver(int solver_code = static_cast<int>(veqlib_kernel_api::SolverMethodPowell))
            : solver_(solver_kind_from_runtime_method_code(solver_code)), context_(make_context(solver_))
        {
        }

        nb::dict metadata() const { return topology_metadata_dict(context_->input); }

        std::string metadata_json() const { return case_prefix_json(*context_).dump(2); }

        void set_case_json(const std::string& payload)
        {
            if (payload.empty())
            {
                last_case_json_ = "{}";
                return;
            }
            const nlohmann::json data = nlohmann::json::parse(payload);
            if (!data.is_object())
                throw std::runtime_error("KernelSolver.set_case_json expects a JSON object");
            if (data.empty())
            {
                last_case_json_ = data.dump();
                return;
            }

            CaseInput next_input = case_input_from_json(data, solver_);
            apply_runtime_case(std::move(next_input), data.dump());
        }

        void set_kernel_runtime(const std::string& case_name,
                                double             a,
                                double             R0,
                                double             Z0,
                                double             B0,
                                double             ka,
                                RuntimeArrayView   c_offsets,
                                RuntimeArrayView   s_offsets,
                                RuntimeArrayView   scaled_heat,
                                RuntimeArrayView   scaled_current,
                                double             scaled_Ip,
                                double             beta,
                                double             fix_rho,
                                int                method_code,
                                double             max_residual,
                                int                max_evaluations,
                                double             accepted_residual_factor,
                                double             accepted_residual_floor,
                                int                initial_policy_code,
                                int                residual_normalization_code,
                                double             residual_normalization_floor,
                                double             residual_normalization_max_ratio,
                                double             residual_normalization_huber_tau,
                                int                residual_normalization_probe_count,
                                double             residual_normalization_probe_step,
                                double             residual_normalization_sensitivity_lambda)
        {
            CaseInput next_input = case_input_from_runtime(case_name,
                                                           a,
                                                           R0,
                                                           Z0,
                                                           B0,
                                                           ka,
                                                           c_offsets,
                                                           s_offsets,
                                                           scaled_heat,
                                                           scaled_current,
                                                           scaled_Ip,
                                                           beta,
                                                           fix_rho,
                                                           method_code,
                                                           max_residual,
                                                           max_evaluations,
                                                           accepted_residual_factor,
                                                           accepted_residual_floor,
                                                           initial_policy_code,
                                                           residual_normalization_code,
                                                           residual_normalization_floor,
                                                           residual_normalization_max_ratio,
                                                           residual_normalization_huber_tau,
                                                           residual_normalization_probe_count,
                                                           residual_normalization_probe_step,
                                                           residual_normalization_sensitivity_lambda);
            apply_runtime_case(std::move(next_input), "{}");
        }

        void warmup(size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                SolveResult result = run_solver_once(*context_);
                if (initial_policy_is_warm_clone(context_->input.initial_policy_code))
                    context_->input.x0 = result.x;
            }
        }

        void adopt_last_solution_as_initial()
        {
            if (!has_last_result_)
                throw std::runtime_error("KernelSolver has no solve result to adopt");
            if (!last_result_.accepted || !solver_info_succeeded(context_->input.solver, last_result_.info))
                throw std::runtime_error("KernelSolver cannot adopt an unsuccessful solve result");

            context_->input.x0      = last_result_.x;
            context_->input.x_scale = build_x_block_scale_vector<KernelShape>(
                context_->input.x0,
                profile_params_for_case(context_->input));
            refresh_initial_residual_scale(*context_);
        }

        std::string solve_json()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_       = run_solver_once(*context_);
            has_last_result_   = true;
            if (initial_policy_is_warm_clone(context_->input.initial_policy_code))
                context_->input.x0 = last_result_.x;
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_   = std::chrono::duration<double, std::milli>(elapsed).count();

            const nlohmann::json report = {
                {"schema", "veqlib.kernel.solve_result.v1"},
                {"route", source_route_label()},
                {"x_size", KernelShape::x_size},
                {"solver", solver_json(context_->input)},
                {"elapsed_ms", last_elapsed_ms_},
                {"final", solve_result_json(last_result_)},
                {"success", last_result_.accepted && solver_info_succeeded(context_->input.solver, last_result_.info)},
            };
            return report.dump(2);
        }

        nb::tuple solve_direct()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_       = run_solver_once(*context_);
            has_last_result_   = true;
            if (initial_policy_is_warm_clone(context_->input.initial_policy_code))
                context_->input.x0 = last_result_.x;
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

        void residual_var_into(PackedArrayView x, MutablePackedArrayView out)
        {
            context_->raw_residual(std::span<const double, KernelShape::x_size>{x.data(), KernelShape::x_size},
                                   std::span<double, KernelShape::x_size>{out.data(), KernelShape::x_size});
        }

        double last_elapsed_ms() const noexcept { return last_elapsed_ms_; }

    private:
        void apply_runtime_case(CaseInput next_input, std::string last_case_json)
        {
            if (initial_policy_is_warm_clone(next_input.initial_policy_code))
            {
                next_input.x0      = context_->input.x0;
                next_input.x_scale = build_x_block_scale_vector<KernelShape>(
                    next_input.x0,
                    profile_params_for_case(next_input));
            }
            auto next_context = std::make_unique<SolveContext>(next_input);
            refine_cold_initial_state(*next_context);
            refresh_initial_residual_scale(*next_context);
            context_        = std::move(next_context);
            last_case_json_ = std::move(last_case_json);
        }

        SolverKind                    solver_;
        std::unique_ptr<SolveContext> context_;
        SolveResult                   last_result_{};
        bool                          has_last_result_ = false;
        std::string                   last_case_json_  = "{}";
        double                        last_elapsed_ms_ = 0.0;
    };

} // namespace veqlib_python
