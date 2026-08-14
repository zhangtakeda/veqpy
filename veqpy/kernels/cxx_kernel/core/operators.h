#pragma once

// Fused profile, geometry, source, and residual pipeline for generated Cxx Kernel artifacts.

#include "abi_enums.h"
#include "geometry.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"
#include <cstddef>
#include <limits>
#include <span>

namespace operators::detail
{
    using std::size_t;

    using geometry::GeometryRuntime;
    using profiles::ProfileRuntimeParams;
    using profiles::RuntimeProfiles;
    using residual::ResidualRuntime;
    using source::NativeSourceRuntime;
    using source::axis_fix_count;
    using tensor::Vector;

    template <typename Shape, typename SourceShape>
    struct OperatorSetup
    {
        ProfileRuntimeParams<Shape>               profile_params{};
        Vector<double, SourceShape::sample_count> pprime{};
        Vector<double, SourceShape::sample_count> driver{};
    };

    struct OperatorRuntimeScalars
    {
        double a    = 1.0;
        double R0   = 1.0;
        double Z0   = 0.0;
        double B0   = 1.0;
        double p0   = 0.0;
        double Ip   = 0.0;
        double beta = 0.0;
    };

    template <typename Shape,
              typename GridType,
              typename SourceShape,
              int SourceRouteCode           = source_route_pf,
              int SourceConstraintCode      = source_constraint_ip,
              int SourceCoordinateCode      = source_coordinate_psin,
              int SourceNodesCode           = source_nodes_uniform,
              int SourceActiveFamilyCode    = source_active_psin,
              int SourceParameterizationCode = source_parameterization_identity>
    struct SourceOperator
    {
        static_assert(Shape::L_max == GridType::basis_rows, "operator/profile basis rows must match");
        static_assert(Shape::K_max == GridType::r_power_rows, "operator/profile r rows must match");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "operator/profile harmonics must match");
        static_assert(SourceRouteCode == source_route_pf || SourceRouteCode == source_route_pp ||
                          SourceRouteCode == source_route_pi || SourceRouteCode == source_route_pj1 ||
                          SourceRouteCode == source_route_pj2 || SourceRouteCode == source_route_pj3 ||
                          SourceRouteCode == source_route_pq,
                      "native source topology currently supports PF, PP, PI, PJ1, PJ2, PJ3, and PQ routes");
        static_assert((SourceRouteCode == source_route_pf &&
                       (SourceConstraintCode == source_constraint_null || SourceConstraintCode == source_constraint_ip ||
                        SourceConstraintCode == source_constraint_beta)) ||
                          ((SourceRouteCode == source_route_pp || SourceRouteCode == source_route_pi ||
                            SourceRouteCode == source_route_pj1 || SourceRouteCode == source_route_pj2 ||
                            SourceRouteCode == source_route_pj3 || SourceRouteCode == source_route_pq) &&
                           (SourceConstraintCode == source_constraint_null || SourceConstraintCode == source_constraint_ip ||
                            SourceConstraintCode == source_constraint_beta ||
                            SourceConstraintCode == source_constraint_ip_beta)),
                      "source topology constraint is not implemented for this native route");
        static_assert(SourceCoordinateCode == source_coordinate_r || SourceCoordinateCode == source_coordinate_psin,
                      "native source topology supports r or psin coordinates");
        static_assert(SourceNodesCode == source_nodes_uniform || SourceNodesCode == source_nodes_grid,
                      "native source topology supports uniform or grid nodes");
        static_assert(SourceActiveFamilyCode == source_active_none || SourceActiveFamilyCode == source_active_psin ||
                          SourceActiveFamilyCode == source_active_F,
                      "native source topology supports no active source family, active psin, or active F ownership");
        static_assert(SourceParameterizationCode == source_parameterization_identity ||
                          SourceParameterizationCode == source_parameterization_sqrt_psin,
                      "native source topology received an unsupported source parameterization");
        static_assert(SourceParameterizationCode == source_parameterization_identity ||
                          (SourceRouteCode == source_route_pp &&
                           SourceCoordinateCode == source_coordinate_psin &&
                           SourceNodesCode == source_nodes_uniform),
                      "sqrt_psin parameterization is only implemented for PP/psin/uniform");
        static_assert(
            SourceActiveFamilyCode != source_active_psin ||
                (SourceCoordinateCode == source_coordinate_psin && SourceNodesCode == source_nodes_uniform),
            "active psin ownership is only implemented for psin/uniform source routes");
        static_assert(
            SourceActiveFamilyCode == source_active_psin || SourceRouteCode == source_route_pj2 ||
                SourceRouteCode == source_route_pj3 ||
                !(SourceCoordinateCode == source_coordinate_psin && SourceNodesCode == source_nodes_uniform),
            "psin/uniform source routes require active psin ownership");
        static_assert(SourceActiveFamilyCode != source_active_F ||
                          SourceRouteCode == source_route_pj2 || SourceRouteCode == source_route_pj3,
                      "active F ownership is only implemented for PJ2/PJ3 source topology");
        static_assert((SourceRouteCode != source_route_pj2 && SourceRouteCode != source_route_pj3) ||
                          SourceActiveFamilyCode == source_active_F,
                      "PJ2/PJ3 source topology requires active F ownership");
        static_assert(SourceActiveFamilyCode != source_active_psin ||
                          Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "profile-owned source topology requires an active psin profile");
        static_assert(SourceActiveFamilyCode == source_active_psin ||
                          !Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "source-owned source topology does not accept an active psin profile");
        static_assert(SourceActiveFamilyCode != source_active_F ||
                          Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "PJ2/PJ3 source topology requires an active F profile");
        static_assert(SourceActiveFamilyCode == source_active_F ||
                          !Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "non-PJ2/PJ3 source topology does not accept an active F profile");
        static_assert(SourceNodesCode != source_nodes_grid || SourceShape::sample_count == GridType::radial_nodes,
                      "grid source nodes require source samples to match radial nodes");
        static_assert(SourceShape::sample_count >= 1, "source topology needs at least one sample");

        using shape        = Shape;
        using grid         = GridType;
        using source_shape = SourceShape;
        using Setup        = OperatorSetup<Shape, SourceShape>;
        using RuntimeScalars  = OperatorRuntimeScalars;
        using Profiles     = RuntimeProfiles<Shape, GridType>;
        using Geometry     = GeometryRuntime<GridType>;
        using Source       = NativeSourceRuntime<GridType, SourceShape>;
        using Residual     = ResidualRuntime<Shape, GridType>;
        using PackedVector = typename Residual::PackedVector;

        struct OperatorPlan
        {
            Profiles                    fixed_profiles{};
            ProfileRuntimeParams<Shape> profile_params{};
            size_t                      n_axis_fix = 0;
        };

        struct OperatorWorkspace
        {
            Profiles profiles{};
            Geometry geometry{};
            Source   source_runtime{};
            Residual residual{};
        };

        explicit constexpr SourceOperator(const Setup& setup) noexcept : plan(make_plan(setup))
        {
            workspace.profiles.load_fixed_from(plan.fixed_profiles);
            workspace.source_runtime.set_uniform_sources(source_span(setup.pprime), source_span(setup.driver));
        }

        constexpr const RuntimeScalars& runtime_scalars() const noexcept { return runtime_scalars_; }

        constexpr void set_runtime_scalars(const RuntimeScalars& params) noexcept { runtime_scalars_ = params; }

        constexpr void reprepare(const Setup& setup) noexcept
        {
            plan = make_plan(setup);
            workspace.profiles.load_fixed_from(plan.fixed_profiles);
            workspace.source_runtime.set_uniform_sources(source_span(setup.pprime), source_span(setup.driver));
        }

        static constexpr void evaluate_impl(const OperatorPlan&                       plan,
                                            const RuntimeScalars&                      runtime_scalars,
                                            OperatorWorkspace&                        workspace,
                                            std::span<const double, Shape::x_size>     x,
                                            std::span<double, Shape::x_size>           out,
                                            const double*                              output_scale) noexcept
        {
            workspace.profiles.refresh_active(x, plan.profile_params);
            workspace.geometry.update(runtime_scalars.a, runtime_scalars.R0, runtime_scalars.Z0, workspace.profiles);

            const auto materialize_source = [&]() constexpr {
                if constexpr (SourceActiveFamilyCode == source_active_psin)
                {
                    workspace.source_runtime.template materialize_profile_owned_psin<SourceParameterizationCode>(
                        workspace.profiles, plan.n_axis_fix);
                }
                else
                {
                    if constexpr ((SourceRouteCode == source_route_pj2 || SourceRouteCode == source_route_pj3) &&
                                  SourceCoordinateCode == source_coordinate_psin &&
                                  SourceNodesCode == source_nodes_uniform)
                    {
                        // PJ2/PJ3 remap their effective source samples inside
                        // the coupled psin/current fixed-point loop.
                    }
                    else if constexpr (SourceNodesCode == source_nodes_grid)
                        workspace.source_runtime.materialize_grid_sources();
                    else
                        workspace.source_runtime.materialize_r_uniform_sources();
                }
                if constexpr (SourceActiveFamilyCode == source_active_F)
                    workspace.source_runtime.materialize_active_F(workspace.profiles);
            };

            const auto update_source = [&]() constexpr {
            if constexpr (SourceRouteCode == source_route_pf)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pf_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pf_psin_uniform<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pp)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pp_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pp_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pi)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pi_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pi_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pj1)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pj1_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pj1_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pj2)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pj2_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else if constexpr (SourceNodesCode == source_nodes_uniform)
                    workspace.source_runtime.template update_pj2_psin_uniform_fixed_point<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pj2_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pj3)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pj3_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else if constexpr (SourceNodesCode == source_nodes_uniform)
                    workspace.source_runtime.template update_pj3_psin_uniform_fixed_point<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pj3_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }
            else
            {
                if constexpr (SourceCoordinateCode == source_coordinate_r)
                    workspace.source_runtime.template update_pq_r<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pq_psin<SourceConstraintCode>(
                        workspace.geometry,
                        runtime_scalars.R0,
                        runtime_scalars.p0,
                        runtime_scalars.Ip,
                        runtime_scalars.beta,
                        runtime_scalars.B0,
                        plan.n_axis_fix);
            }

            };

            if constexpr (SourceCoordinateCode == source_coordinate_psin)
            {
                // Public P_psin and, for PF, FF_psin satisfy the same alpha2
                // chain rule with the conventional P_psi/FF_psi consumed by
                // route kernels. Close both relations in one scalar iteration.
                double alpha2_guess = 1.0;
                bool   converged    = false;
                for (size_t iteration = 0; iteration < 64; ++iteration)
                {
                    if (!math::is_finite(alpha2_guess) || math::abs(alpha2_guess) <= 1.0e-14)
                        break;
                    const double derivative_scale =
                        SourceRouteCode == source_route_pf ? math::abs(alpha2_guess) : alpha2_guess;
                    workspace.source_runtime.set_pressure_input_scale(1.0 / derivative_scale);
                    if constexpr (SourceRouteCode == source_route_pf)
                        workspace.source_runtime.set_driver_input_scale(1.0 / derivative_scale);
                    materialize_source();
                    update_source();
                    const double next = workspace.source_runtime.alpha2;
                    const double next_scale =
                        SourceRouteCode == source_route_pf ? math::abs(next) : next;
                    const double defect =
                        math::abs(next_scale - alpha2_guess) / math::max(math::abs(next_scale), 1.0e-14);
                    if (defect <= 1.0e-10)
                    {
                        converged = true;
                        break;
                    }
                    // PF with both coordinate derivatives scaled has the
                    // homogeneous null-constraint map g(a)~C/a. Equal-weight
                    // averaging removes its direct-Picard two-cycle. psin is
                    // axis-to-edge oriented, so PF closes |alpha2| while Ip
                    // retains ownership of the returned flux-direction sign.
                    if constexpr (SourceRouteCode == source_route_pf)
                        alpha2_guess = 0.5 * (alpha2_guess + math::abs(next));
                    else
                        alpha2_guess = next;
                }
                if (!converged)
                {
                    for (double& value : out)
                        value = std::numeric_limits<double>::quiet_NaN();
                    return;
                }
            }
            else
            {
                workspace.source_runtime.set_pressure_input_scale(1.0);
                workspace.source_runtime.set_driver_input_scale(1.0);
                materialize_source();
                update_source();
            }

            workspace.source_runtime.template finalize_pressure_normalization<
                SourceCoordinateCode == source_coordinate_r>(
                runtime_scalars.p0,
                SourceConstraintCode == source_constraint_beta ||
                    SourceConstraintCode == source_constraint_ip_beta);

            if constexpr (SourceActiveFamilyCode == source_active_none || SourceActiveFamilyCode == source_active_F)
                workspace.source_runtime.publish_source_target_root_fields();

            workspace.residual.update_compact(workspace.source_runtime, workspace.geometry);
            if (output_scale == nullptr)
                workspace.residual.pack_into(out, runtime_scalars.a, runtime_scalars.R0, runtime_scalars.B0);
            else
                workspace.residual.pack_scaled_into(
                    out, runtime_scalars.a, runtime_scalars.R0, runtime_scalars.B0, output_scale);
        }

        static constexpr void evaluate_with(const OperatorPlan&                       plan,
                                            const RuntimeScalars&                      runtime_scalars,
                                            OperatorWorkspace&                        workspace,
                                            std::span<const double, Shape::x_size>     x,
                                            PackedVector&                              out) noexcept
        {
            evaluate_impl(plan, runtime_scalars, workspace, x, out.span(), nullptr);
        }

        static constexpr void evaluate_scaled_with(const OperatorPlan&                   plan,
                                                   const RuntimeScalars&                  runtime_scalars,
                                                   OperatorWorkspace&                    workspace,
                                                   std::span<const double, Shape::x_size> x,
                                                   std::span<double, Shape::x_size>       out,
                                                   const double*                          output_scale) noexcept
        {
            evaluate_impl(plan, runtime_scalars, workspace, x, out, output_scale);
        }

        constexpr void evaluate(std::span<const double, Shape::x_size> x, PackedVector& out) noexcept
        {
            evaluate_with(plan, runtime_scalars_, workspace, x, out);
        }

        constexpr void evaluate_scaled(std::span<const double, Shape::x_size> x,
                                       std::span<double, Shape::x_size>       out,
                                       const double*                          output_scale) noexcept
        {
            evaluate_scaled_with(plan, runtime_scalars_, workspace, x, out, output_scale);
        }

        OperatorPlan      plan{};
        OperatorWorkspace workspace{};

    private:
        static constexpr OperatorPlan make_plan(const Setup& setup) noexcept
        {
            OperatorPlan out{};
            out.profile_params = setup.profile_params;
            out.n_axis_fix     = axis_fix_count<GridType>();
            out.fixed_profiles.refresh_fixed(out.profile_params);
            return out;
        }

        static constexpr std::span<const double, SourceShape::sample_count>
        source_span(const Vector<double, SourceShape::sample_count>& values) noexcept
        {
            return std::span<const double, SourceShape::sample_count>{values.data(), SourceShape::sample_count};
        }

        RuntimeScalars runtime_scalars_{};
    };
} // namespace operators::detail

namespace operators
{
    using detail::SourceOperator;
    using detail::OperatorSetup;
    using detail::OperatorRuntimeScalars;
} // namespace operators
