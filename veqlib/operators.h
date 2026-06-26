#pragma once

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
        double                                    fix_rho = 0.0;
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

    template <typename Shape, typename GridType, typename SourceShape, int SourceConstraintCode = 1>
    struct PfPsinUniformIpOperator
    {
        static_assert(Shape::L_max == GridType::basis_rows, "operator/profile basis rows must match");
        static_assert(Shape::K_max == GridType::rho_power_rows, "operator/profile rho rows must match");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "operator/profile harmonics must match");
        static_assert(Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "PF/psin/uniform requires an active psin profile");
        static_assert(!Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "PF/psin/uniform does not accept an active F profile");
        static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2,
                      "PF/psin/uniform supports null, Ip, or beta constraints");
        static_assert(SourceShape::sample_count >= 1, "PF/psin/uniform source needs at least one sample");

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

            workspace.source_runtime.materialize_profile_owned_psin(workspace.profiles, plan.n_axis_fix);
            workspace.source_runtime.template update_pf_psin_uniform<SourceConstraintCode>(
                workspace.geometry,
                solve_params.Ip,
                solve_params.beta,
                solve_params.B0,
                plan.n_axis_fix);

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
            out.n_axis_fix     = axis_fix_count<GridType>(setup.fix_rho);
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
