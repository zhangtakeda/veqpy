#pragma once

#include "geometry.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include <cstddef>
#include <span>

namespace operator_pf::detail
{
    using std::size_t;

    template <typename Shape>
    struct PfPsinUniformRuntimeParams
    {
        profiles::ProfileRuntimeParams<Shape> profile_params{};
        double                                a       = 1.0;
        double                                R0      = 1.0;
        double                                Z0      = 0.0;
        double                                B0      = 1.0;
        double                                Ip      = 0.0;
        double                                fix_rho = 0.0;
    };

    template <typename Shape, typename GridType, typename SourceShape>
    struct PfPsinUniformOperator
    {
        static_assert(Shape::L_max == GridType::basis_rows, "operator/profile basis rows must match");
        static_assert(Shape::K_max == GridType::rho_power_rows, "operator/profile rho rows must match");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "operator/profile harmonics must match");
        static_assert(Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                      "PF/psin/uniform requires an active psin profile");
        static_assert(!Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                      "PF/psin/uniform does not accept an active F profile");
        static_assert(SourceShape::sample_count >= 1, "PF/psin/uniform source needs at least one sample");

        using shape         = Shape;
        using grid          = GridType;
        using source_shape  = SourceShape;
        using RuntimeParams = PfPsinUniformRuntimeParams<Shape>;
        using Profiles      = profiles::RuntimeProfiles<Shape, GridType>;
        using Geometry      = geometry::GeometryRuntime<GridType>;
        using Source        = source::ProfileOwnedPsinSourceRuntime<GridType, SourceShape>;
        using Residual      = residual::ResidualRuntime<Shape, GridType>;
        using PackedVector  = typename Residual::PackedVector;

        struct KernelPlan
        {
            Profiles      fixed_profiles{};
            RuntimeParams static_params{};
            size_t        n_axis_fix = 0;
            bool          prepared   = false;

            constexpr void refresh(const RuntimeParams& params) noexcept
            {
                static_params.profile_params = params.profile_params;
                static_params.fix_rho        = params.fix_rho;
                n_axis_fix                   = source::axis_fix_count<GridType>(params.fix_rho);
                fixed_profiles.refresh_fixed(static_params.profile_params);
                prepared = true;
            }
        };

        struct KernelWorkspace
        {
            Profiles profiles{};
            Geometry geometry{};
            Source   source_runtime{};
            Residual residual{};
        };

        KernelPlan      plan{};
        KernelWorkspace workspace{};

        constexpr const RuntimeParams& runtime_params() const noexcept { return params_; }

        constexpr void set_runtime_params(const RuntimeParams& params) noexcept
        {
            params_       = params;
            plan.prepared = false;
        }

        constexpr void set_uniform_sources(std::span<const double, SourceShape::sample_count> heat,
                                           std::span<const double, SourceShape::sample_count> current) noexcept
        {
            workspace.source_runtime.set_uniform_sources(heat, current);
        }

        constexpr void refresh_static_plan() noexcept
        {
            plan.refresh(params_);
            workspace.profiles.load_fixed_from(plan.fixed_profiles);
        }

        constexpr void evaluate(std::span<const double, Shape::x_size> x, PackedVector& out) noexcept
        {
            if (!plan.prepared)
                refresh_static_plan();
            workspace.profiles.refresh_active(x, params_.profile_params);
            workspace.geometry.update(params_.a, params_.R0, params_.Z0, workspace.profiles);

            workspace.source_runtime.materialize_profile_owned_psin(workspace.profiles, plan.n_axis_fix);
            workspace.source_runtime.update_pf_ip_from_psin_uniform(workspace.geometry, params_.Ip, plan.n_axis_fix);

            workspace.residual.update_compact(workspace.source_runtime, workspace.geometry);
            workspace.residual.pack_into(out, params_.a, params_.R0, params_.B0);
        }

    private:
        RuntimeParams params_{};
    };
} // namespace operator_pf::detail

namespace operator_pf
{
    using detail::PfPsinUniformOperator;
    using detail::PfPsinUniformRuntimeParams;
} // namespace operator_pf
