#pragma once

#include "geometry.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"
#include <span>

namespace operator_pf::detail
{
    template <typename Shape>
    struct PfPsinUniformRuntimeParams
    {
        profiles::ProfileRuntimeParams<Shape> profile_params{};
        double a       = 1.0;
        double R0      = 1.0;
        double Z0      = 0.0;
        double B0      = 1.0;
        double Ip      = 0.0;
        double fix_rho = 0.0;
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

        using shape          = Shape;
        using grid           = GridType;
        using source_shape   = SourceShape;
        using RuntimeParams  = PfPsinUniformRuntimeParams<Shape>;
        using Profiles       = profiles::RuntimeProfiles<Shape, GridType>;
        using Geometry       = geometry::GeometryRuntime<GridType>;
        using Source         = source::ProfileOwnedPsinSourceRuntime<GridType, SourceShape>;
        using Residual       = residual::ResidualRuntime<Shape, GridType>;
        using PackedVector   = typename Residual::PackedVector;

        Profiles profiles{};
        Geometry geometry{};
        Source   source_runtime{};
        Residual residual{};
        RuntimeParams params{};

        constexpr void set_uniform_sources(std::span<const double, SourceShape::sample_count> heat,
                                           std::span<const double, SourceShape::sample_count> current) noexcept
        {
            source_runtime.set_uniform_sources(heat, current);
        }

        constexpr void evaluate(std::span<const double, Shape::x_size> x, PackedVector& out) noexcept
        {
            profiles.refresh_fixed(params.profile_params);
            profiles.refresh_active(x, params.profile_params);
            geometry.update(params.a, params.R0, params.Z0, profiles);

            const auto n_axis_fix = source::axis_fix_count<GridType>(params.fix_rho);
            source_runtime.materialize_profile_owned_psin(profiles, n_axis_fix);
            source_runtime.update_pf_ip_from_psin_uniform(geometry, params.Ip, n_axis_fix);

            residual.update_compact(source_runtime, geometry);
            residual.pack_into(out, params.a, params.R0, params.B0);
        }
    };
} // namespace operator_pf::detail

namespace operator_pf
{
    using detail::PfPsinUniformOperator;
    using detail::PfPsinUniformRuntimeParams;
} // namespace operator_pf
