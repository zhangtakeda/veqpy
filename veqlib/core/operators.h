#pragma once

#include "abi_enums.h"
#include "geometry.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"
#include <cstddef>
#include <span>

namespace operators::detail
{
    using std::size_t;

    using geometry::GeometryRuntime;
    using profiles::ProfileRuntimeParams;
    using profiles::RuntimeProfiles;
    using residual::ResidualRuntime;
    using source::PfPsinUniformIpSourceRuntime;
    using source::axis_fix_count;
    using tensor::Vector;

    template <typename Shape, typename SourceShape>
    struct PfPsinUniformIpSetup
    {
        ProfileRuntimeParams<Shape>               profile_params{};
        Vector<double, SourceShape::sample_count> heat{};
        Vector<double, SourceShape::sample_count> current{};
    };

    struct PfPsinUniformIpSolveParams
    {
        double a    = 1.0;
        double R0   = 1.0;
        double Z0   = 0.0;
        double B0   = 1.0;
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
    struct PfPsinUniformIpOperator
    {
        static_assert(Shape::L_max == GridType::basis_rows, "operator/profile basis rows must match");
        static_assert(Shape::K_max == GridType::rho_power_rows, "operator/profile rho rows must match");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "operator/profile harmonics must match");
        static_assert(SourceRouteCode == source_route_pf || SourceRouteCode == source_route_pp ||
                          SourceRouteCode == source_route_pi || SourceRouteCode == source_route_pj1 ||
                          SourceRouteCode == source_route_pj2 || SourceRouteCode == source_route_pq,
                      "native source topology currently supports PF, PP, PI, PJ1, PJ2, and PQ routes");
        static_assert((SourceRouteCode == source_route_pf &&
                       (SourceConstraintCode == source_constraint_null || SourceConstraintCode == source_constraint_ip ||
                        SourceConstraintCode == source_constraint_beta)) ||
                          ((SourceRouteCode == source_route_pp || SourceRouteCode == source_route_pi ||
                            SourceRouteCode == source_route_pj1 || SourceRouteCode == source_route_pj2 ||
                            SourceRouteCode == source_route_pq) &&
                           (SourceConstraintCode == source_constraint_null || SourceConstraintCode == source_constraint_ip ||
                            SourceConstraintCode == source_constraint_beta ||
                            SourceConstraintCode == source_constraint_ip_beta)),
                      "source topology constraint is not implemented for this native route");
        static_assert(SourceCoordinateCode == source_coordinate_rho || SourceCoordinateCode == source_coordinate_psin,
                      "native source topology supports rho or psin coordinates");
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
                !(SourceCoordinateCode == source_coordinate_psin && SourceNodesCode == source_nodes_uniform),
            "psin/uniform source routes require active psin ownership");
        static_assert(SourceActiveFamilyCode != source_active_F || SourceRouteCode == source_route_pj2,
                      "active F ownership is only implemented for PJ2 source topology");
        static_assert(SourceRouteCode != source_route_pj2 || SourceActiveFamilyCode == source_active_F,
                      "PJ2 source topology requires active F ownership");
        static_assert(SourceActiveFamilyCode != source_active_psin ||
                          Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "profile-owned source topology requires an active psin profile");
        static_assert(SourceActiveFamilyCode == source_active_psin ||
                          !Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "source-owned source topology does not accept an active psin profile");
        static_assert(SourceActiveFamilyCode != source_active_F ||
                          Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "PJ2 source topology requires an active F profile");
        static_assert(SourceActiveFamilyCode == source_active_F ||
                          !Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "non-PJ2 source topology does not accept an active F profile");
        static_assert(SourceNodesCode != source_nodes_grid || SourceShape::sample_count == GridType::radial_nodes,
                      "grid source nodes require source samples to match radial nodes");
        static_assert(SourceShape::sample_count >= 1, "source topology needs at least one sample");

        using shape        = Shape;
        using grid         = GridType;
        using source_shape = SourceShape;
        using Setup        = PfPsinUniformIpSetup<Shape, SourceShape>;
        using SolveParams  = PfPsinUniformIpSolveParams;
        using Profiles     = RuntimeProfiles<Shape, GridType>;
        using Geometry     = GeometryRuntime<GridType>;
        using Source       = PfPsinUniformIpSourceRuntime<GridType, SourceShape>;
        using Residual     = ResidualRuntime<Shape, GridType>;
        using PackedVector = typename Residual::PackedVector;

        struct KernelPlan
        {
            Profiles                    fixed_profiles{};
            ProfileRuntimeParams<Shape> profile_params{};
            size_t                      n_axis_fix = 0;
        };

        struct KernelWorkspace
        {
            Profiles profiles{};
            Geometry geometry{};
            Source   source_runtime{};
            Residual residual{};
        };

        explicit constexpr PfPsinUniformIpOperator(const Setup& setup) noexcept : plan(make_plan(setup))
        {
            workspace.profiles.load_fixed_from(plan.fixed_profiles);
            workspace.source_runtime.set_uniform_sources(source_span(setup.heat), source_span(setup.current));
        }

        constexpr const SolveParams& solve_params() const noexcept { return solve_params_; }

        constexpr void set_solve_params(const SolveParams& params) noexcept { solve_params_ = params; }

        constexpr void reprepare(const Setup& setup) noexcept
        {
            plan = make_plan(setup);
            workspace.profiles.load_fixed_from(plan.fixed_profiles);
            workspace.source_runtime.set_uniform_sources(source_span(setup.heat), source_span(setup.current));
        }

        static constexpr void evaluate_with(const KernelPlan&    plan,
                                            const SolveParams&   solve_params,
                                            KernelWorkspace&     workspace,
                                            std::span<const double, Shape::x_size> x,
                                            PackedVector& out) noexcept
        {
            workspace.profiles.refresh_active(x, plan.profile_params);
            workspace.geometry.update(solve_params.a, solve_params.R0, solve_params.Z0, workspace.profiles);

            if constexpr (SourceActiveFamilyCode == source_active_psin)
            {
                workspace.source_runtime.template materialize_profile_owned_psin<SourceParameterizationCode>(
                    workspace.profiles, plan.n_axis_fix);
            }
            else
            {
                if constexpr (SourceRouteCode == source_route_pj2 &&
                              SourceCoordinateCode == source_coordinate_psin &&
                              SourceNodesCode == source_nodes_uniform)
                {
                    // PJ2/psin/uniform remaps source samples inside its fixed-point source loop.
                }
                else if constexpr (SourceNodesCode == source_nodes_grid)
                    workspace.source_runtime.materialize_grid_sources();
                else
                    workspace.source_runtime.materialize_rho_uniform_sources();
            }
            if constexpr (SourceActiveFamilyCode == source_active_F)
                workspace.source_runtime.materialize_active_F(workspace.profiles);

            if constexpr (SourceRouteCode == source_route_pf)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pf_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pf_psin_uniform<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pp)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pp_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pp_psin<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pi)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pi_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pi_psin<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pj1)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pj1_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pj1_psin<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }
            else if constexpr (SourceRouteCode == source_route_pj2)
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pj2_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.R0,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else if constexpr (SourceNodesCode == source_nodes_uniform)
                    workspace.source_runtime.template update_pj2_psin_uniform_fixed_point<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.R0,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pj2_psin<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.R0,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }
            else
            {
                if constexpr (SourceCoordinateCode == source_coordinate_rho)
                    workspace.source_runtime.template update_pq_rho<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.R0,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
                else
                    workspace.source_runtime.template update_pq_psin<SourceConstraintCode>(
                        workspace.geometry,
                        solve_params.R0,
                        solve_params.Ip,
                        solve_params.beta,
                        solve_params.B0,
                        plan.n_axis_fix);
            }

            if constexpr (SourceActiveFamilyCode == source_active_none || SourceActiveFamilyCode == source_active_F)
                workspace.source_runtime.publish_source_target_root_fields();

            workspace.residual.update_compact(workspace.source_runtime, workspace.geometry);
            workspace.residual.pack_into(out, solve_params.a, solve_params.R0, solve_params.B0);
        }

        constexpr void evaluate(std::span<const double, Shape::x_size> x, PackedVector& out) noexcept
        {
            evaluate_with(plan, solve_params_, workspace, x, out);
        }

        KernelPlan      plan{};
        KernelWorkspace workspace{};

    private:
        static constexpr KernelPlan make_plan(const Setup& setup) noexcept
        {
            KernelPlan out{};
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

        SolveParams solve_params_{};
    };
} // namespace operators::detail

namespace operators
{
    using detail::PfPsinUniformIpOperator;
    using detail::PfPsinUniformIpSetup;
    using detail::PfPsinUniformIpSolveParams;
} // namespace operators
