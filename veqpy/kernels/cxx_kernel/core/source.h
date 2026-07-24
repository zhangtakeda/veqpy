#pragma once

// Source route materialization and scaling helpers for generated Cxx Kernel artifacts.

#include "geometry.h"
#include "linalg.h"
#include "veq_numeric.h"
#include "tensor.h"
#include "tensor_kernels.h"
#include "tensor_layout.h"
#include <cstddef>
#include <limits>
#include <span>

namespace source::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;
    using tensor_kernels::matvec_into;
    using tensor_kernels::multi_matvec_into;
    using tensor_layout::RadialGridMatvecPlan;

    inline constexpr size_t default_barycentric_stencil = 8;

    inline constexpr size_t root_psin        = 0;
    inline constexpr size_t root_psin_r      = 1;
    inline constexpr size_t root_psin_rr     = 2;
    inline constexpr size_t root_field_count = 3;

    inline constexpr size_t profile_value   = 0;
    inline constexpr size_t profile_radial  = 1;
    inline constexpr size_t profile_radial2 = 2;

    inline constexpr size_t pj2_psin_uniform_fixed_point_max_iter = 16;
    inline constexpr double pj2_psin_uniform_fixed_point_max_residual = 1.0e-10;
    // Fixed axis cutoff: radial nodes with rho below this value use axis-regularized profiles.
    inline constexpr double axis_fix_rho = 0.05;

    constexpr size_t clipped_stencil_size(size_t sample_count) noexcept
    {
        return sample_count < default_barycentric_stencil ? sample_count : default_barycentric_stencil;
    }

    template <size_t SampleCount, size_t StencilSize = clipped_stencil_size(SampleCount)>
    struct SampledSourceShape
    {
        static_assert(SampleCount >= 1, "source samples require at least one sample");
        static_assert(StencilSize >= 1, "source barycentric stencil must be positive");
        static_assert(StencilSize <= SampleCount, "source barycentric stencil exceeds sample count");

        static constexpr size_t sample_count = SampleCount;
        static constexpr size_t stencil_size = StencilSize;

        static consteval auto make_barycentric_weights()
        {
            Vector<double, stencil_size> weights{uninitialized};
            weights[0] = 1.0;
            for (size_t j = 1; j < stencil_size; ++j)
                weights[j] = -weights[j - 1] * static_cast<double>(stencil_size - j) / static_cast<double>(j);
            return weights;
        }

        static constexpr auto barycentric_weights = make_barycentric_weights();
    };

    template <typename GridType>
    constexpr size_t axis_fix_count() noexcept
    {
        size_t count = 0;
        while (count < GridType::radial_nodes && GridType::nodes[count] < axis_fix_rho)
            ++count;
        return count;
    }

    template <typename GridType>
    constexpr void radial_grid_multi_matvec_into(Vector<double, GridType::radial_nodes>&       out0,
                                                 Vector<double, GridType::radial_nodes>&       out1,
                                                 const Vector<double, GridType::radial_nodes>& values) noexcept
    {
        using Plan = RadialGridMatvecPlan<GridType>;
        multi_matvec_into(out0, out1, Plan::derivative_accumulator, values);
    }

    template <typename GridType>
    constexpr void radial_grid_accumulator_matvec_into(Vector<double, GridType::radial_nodes>&       out,
                                                       const Vector<double, GridType::radial_nodes>& values) noexcept
    {
        using Plan = RadialGridMatvecPlan<GridType>;
        matvec_into(out, Plan::accumulator, values);
    }

    template <typename GridType, typename SourceShape>
    struct NativeSourceRuntime
    {
        static constexpr size_t radial_nodes = GridType::radial_nodes;
        static constexpr size_t sample_count = SourceShape::sample_count;
        static constexpr size_t stencil_size = SourceShape::stencil_size;

        using RadialVector = Vector<double, radial_nodes>;
        using SourceVector = Vector<double, sample_count>;
        using RootFields   = Matrix<double, root_field_count, radial_nodes>;

        SourceVector pprime_input{};
        SourceVector driver_input{};
        RadialVector source_psin_query{};
        RadialVector source_parameter_query{};
        RadialVector materialized_pprime_input{};
        RadialVector materialized_driver_input{};
        RootFields   profile_root_fields{};
        RootFields   source_target_root_fields{};
        RadialVector active_F{};
        RadialVector active_F_r{};
        RadialVector FFn_psin{};
        RadialVector Pn_psin{};
        double       alpha1 = 0.0;
        double       alpha2 = 0.0;
        bool         source_materialization_initialized = false;

        constexpr void set_uniform_sources(std::span<const double, sample_count> pprime,
                                           std::span<const double, sample_count> driver) noexcept
        {
            for (size_t i = 0; i < sample_count; ++i)
            {
                pprime_input[i] = pprime[i];
                driver_input[i] = driver[i];
            }
            source_materialization_initialized = false;
        }

        constexpr void materialize_rho_uniform_sources() noexcept
        {
            if (source_materialization_initialized)
                return;
            for (size_t i = 0; i < radial_nodes; ++i)
                source_parameter_query[i] = GridType::nodes[i];
            local_barycentric_interpolate_pair();
            source_materialization_initialized = true;
        }

        constexpr void materialize_grid_sources() noexcept
        {
            static_assert(sample_count == radial_nodes, "grid source inputs must match radial node count");

            if (source_materialization_initialized)
                return;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                materialized_pprime_input[i] = pprime_input[i];
                materialized_driver_input[i] = driver_input[i];
            }
            source_materialization_initialized = true;
        }

        template <int SourceParameterizationCode = 0, typename ProfilesRuntime>
        constexpr void materialize_profile_owned_psin(const ProfilesRuntime& runtime_profiles,
                                                      size_t                 n_axis_fix) noexcept
        {
            static_assert(SourceParameterizationCode == 0 || SourceParameterizationCode == 1,
                          "unsupported profile-owned psin source parameterization");

            using Shape       = typename ProfilesRuntime::shape;
            using ProfileGrid = typename ProfilesRuntime::grid;

            static_assert(ProfileGrid::radial_nodes == radial_nodes, "source/profile radial grids must match");
            static_assert(Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                          "profile-owned psin materialization requires an active psin profile");

            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(root_psin_r, i) =
                    runtime_profiles.profile_field(Shape::psin_profile_id, i, profile_radial);

            regularize_psin_r(n_axis_fix);
            const RadialVector psin_r = const_root_row<root_psin_r>();
            RadialVector       psin_rr{uninitialized};
            RadialVector       integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);
            copy_source_target_to_profile_root();

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double psin_value   = source_target_root_fields(root_psin, i);
                source_psin_query[i]      = psin_value;
                source_parameter_query[i] = psin_value;
                if constexpr (SourceParameterizationCode == 1)
                {
                    if (source_parameter_query[i] < 0.0)
                        source_parameter_query[i] = 0.0;
                    source_parameter_query[i] = math::sqrt(source_parameter_query[i]);
                }
            }

            local_barycentric_interpolate_pair();
        }

        template <typename ProfilesRuntime>
        constexpr void materialize_active_F(const ProfilesRuntime& runtime_profiles) noexcept
        {
            using Shape       = typename ProfilesRuntime::shape;
            using ProfileGrid = typename ProfilesRuntime::grid;

            static_assert(ProfileGrid::radial_nodes == radial_nodes, "source/profile radial grids must match");
            static_assert(Shape::slot_for_profile_id(Shape::F_profile_id).optimized(),
                          "PJ2 source materialization requires an active F profile");

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                active_F[i]   = runtime_profiles.profile_field(Shape::F_profile_id, i, profile_value);
                active_F_r[i] = runtime_profiles.profile_field(Shape::F_profile_id, i, profile_radial);
            }
        }

        constexpr void publish_source_target_root_fields() noexcept { copy_source_target_to_profile_root(); }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pf_rho(const GeometryRuntime& geometry,
                                     double                 p0,
                                     double                 Ip,
                                     double                 beta,
                                     double                 B0,
                                     size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2,
                          "PF source supports null, Ip, or beta constraints");

            RadialVector integrand{uninitialized};
            fill_pf_rho_integrand(integrand, geometry);

            RadialVector psin_r{uninitialized};
            radial_grid_accumulator_matvec_into<GridType>(psin_r, integrand);
            double psi_square_sign       = 1.0;
            double psin_r_weighted_total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                psin_r[i] *= -2.0;
                psin_r_weighted_total += psin_r[i] * GridType::weights[i];
            }

            if (psin_r_weighted_total < 0.0)
            {
                psi_square_sign = -1.0;
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] *= -1.0;
            }

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                if (psin_r[i] < 1.0e-6)
                    psin_r[i] = 1.0e-6;
                psin_r[i] = math::sqrt(psin_r[i]) / geometry.radial_field(geometry::radial_Kn, i);
            }
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            const double integral_prof = dot(psin_r, GridType::weights);
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] /= integral_prof;
            store_root_row<root_psin_r>(psin_r);

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 0)
            {
                alpha2                    = psi_square_sign * integral_prof;
                alpha1                    = -dot(materialized_pprime_input, GridType::weights) / integral_prof;
                alpha1 = ensure_pressure_alpha1<true>(
                    alpha1, p0, alpha2, psin_r);
                const double source_scale = psi_square_sign / (alpha1 * alpha2);
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    Pn_psin[i]  = materialized_pprime_input[i] * source_scale / psin_r[i];
                    FFn_psin[i] = materialized_driver_input[i] * source_scale / psin_r[i];
                }
                regularize_ffn_psin(n_axis_fix);
                (void)Ip;
                (void)beta;
                (void)B0;
                return;
            }

            const double c2 = integral_prof * integral_prof;
            if constexpr (SourceConstraintCode == 1)
            {
                const double G1n_integral = g1n_rho_integral_from_radial_moments(geometry, psi_square_sign);
                alpha1                    = -Ip / G1n_integral;
                (void)beta;
                (void)B0;
            }
            else
            {
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, materialized_pprime_input);
                const double pressure_denominator =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r) +
                    p0 * dot_radial_moment(geometry, geometry::radial_V_r);
                const double numerator =
                    0.5 * beta * B0 * B0 * dot_radial_moment(geometry, geometry::radial_V_r) /
                    pressure_denominator;
                alpha1 = signed_sqrt_ratio(numerator, c2);
                (void)Ip;
            }

            alpha2 = c2 * alpha1;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                Pn_psin[i]  = materialized_pprime_input[i] * psi_square_sign / psin_r[i];
                FFn_psin[i] = materialized_driver_input[i] * psi_square_sign / psin_r[i];
            }
            regularize_ffn_psin(n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pf_psin_uniform(const GeometryRuntime& geometry,
                                              double                 p0,
                                              double                 Ip,
                                              double                 beta,
                                              double                 B0,
                                              size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2,
                          "PF source supports null, Ip, or beta constraints");

            RadialVector integrand{uninitialized};
            fill_pf_psin_integrand(integrand, geometry);

            RadialVector psin_r{uninitialized};
            radial_grid_accumulator_matvec_into<GridType>(psin_r, integrand);
            double psi_scale_sign = 1.0;
            double psin_r_weighted_total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                psin_r[i] *= -1.0;
                psin_r[i] /= geometry.radial_field(geometry::radial_Kn, i);
                psin_r_weighted_total += psin_r[i] * GridType::weights[i];
            }

            if (psin_r_weighted_total < 0.0)
            {
                psi_scale_sign = -1.0;
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] *= -1.0;
            }

            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            const double integral_prof = dot(psin_r, GridType::weights);
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] /= integral_prof;
            store_root_row<root_psin_r>(psin_r);

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 0)
            {
                alpha2 = psi_scale_sign * integral_prof;

                RadialVector pressure_profile{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                    pressure_profile[i] = materialized_pprime_input[i] * psin_r[i];
                alpha1 = -dot(pressure_profile, GridType::weights);
                alpha1 = ensure_pressure_alpha1<false>(
                    alpha1, p0, alpha2, psin_r);

                const double inv_alpha1 = 1.0 / alpha1;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    Pn_psin[i]  = materialized_pprime_input[i] * inv_alpha1;
                    FFn_psin[i] = materialized_driver_input[i] * inv_alpha1;
                }
                regularize_ffn_psin(n_axis_fix);
                (void)Ip;
                (void)beta;
                (void)B0;
                return;
            }

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                Pn_psin[i]  = materialized_pprime_input[i];
                FFn_psin[i] = materialized_driver_input[i];
            }
            regularize_ffn_psin(n_axis_fix);

            if constexpr (SourceConstraintCode == 1)
            {
                const double G1n_integral = g1n_psin_integral_from_radial_moments(geometry);
                alpha1                    = -Ip / G1n_integral;
                alpha2                    = integral_prof * alpha1;
                (void)beta;
                (void)B0;
            }
            else
            {
                RadialVector Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                    Pn_r[i] = Pn_psin[i] * psin_r[i];

                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, Pn_r);
                alpha1 = solve_pf_psin_beta_alpha1(
                    Pn_out,
                    geometry,
                    p0,
                    integral_prof,
                    0.5 * beta * B0 * B0 *
                        dot_radial_moment(geometry, geometry::radial_V_r));
                alpha2 = integral_prof * alpha1;
                (void)Ip;
            }
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pp_rho(const GeometryRuntime& geometry,
                                     double                 p0,
                                     double                 Ip,
                                     double                 beta,
                                     double                 B0,
                                     size_t                 n_axis_fix) noexcept
        {
            update_pp<SourceConstraintCode, true>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pp_psin(const GeometryRuntime& geometry,
                                      double                 p0,
                                      double                 Ip,
                                      double                 beta,
                                      double                 B0,
                                      size_t                 n_axis_fix) noexcept
        {
            update_pp<SourceConstraintCode, false>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pi_rho(const GeometryRuntime& geometry,
                                     double                 p0,
                                     double                 Ip,
                                     double                 beta,
                                     double                 B0,
                                     size_t                 n_axis_fix) noexcept
        {
            update_pi<SourceConstraintCode, true>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pi_psin(const GeometryRuntime& geometry,
                                      double                 p0,
                                      double                 Ip,
                                      double                 beta,
                                      double                 B0,
                                      size_t                 n_axis_fix) noexcept
        {
            update_pi<SourceConstraintCode, false>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pj1_rho(const GeometryRuntime& geometry,
                                      double                 p0,
                                      double                 Ip,
                                      double                 beta,
                                      double                 B0,
                                      size_t                 n_axis_fix) noexcept
        {
            update_pj1<SourceConstraintCode, true>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pj1_psin(const GeometryRuntime& geometry,
                                       double                 p0,
                                       double                 Ip,
                                       double                 beta,
                                       double                 B0,
                                       size_t                 n_axis_fix) noexcept
        {
            update_pj1<SourceConstraintCode, false>(geometry, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pj2_rho(const GeometryRuntime& geometry,
                                      double                 R0,
                                      double                 p0,
                                      double                 Ip,
                                      double                 beta,
                                      double                 B0,
                                      size_t                 n_axis_fix) noexcept
        {
            update_pj2<SourceConstraintCode, true>(geometry, R0, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pj2_psin(const GeometryRuntime& geometry,
                                       double                 R0,
                                       double                 p0,
                                       double                 Ip,
                                       double                 beta,
                                       double                 B0,
                                       size_t                 n_axis_fix) noexcept
        {
            update_pj2<SourceConstraintCode, false>(geometry, R0, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pj2_psin_uniform_fixed_point(const GeometryRuntime& geometry,
                                                           double                 R0,
                                                           double                 p0,
                                                           double                 Ip,
                                                           double                 beta,
                                                           double                 B0,
                                                           size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");

            if (!source_materialization_initialized)
            {
                seed_psin_query_from_passive_psin_profile();
                source_materialization_initialized = true;
            }
            for (size_t iter = 0; iter < pj2_psin_uniform_fixed_point_max_iter; ++iter)
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    source_parameter_query[i] = source_psin_query[i];
                if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
                    local_barycentric_interpolate_pair();
                else
                    local_polynomial_interpolate_pair();

                update_pj2_psin<SourceConstraintCode>(geometry, R0, p0, Ip, beta, B0, n_axis_fix);
                if (update_fixed_point_psin_query())
                    break;
            }

            for (size_t i = 0; i < radial_nodes; ++i)
                source_psin_query[i] = source_target_root_fields(root_psin, i);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pq_rho(const GeometryRuntime& geometry,
                                     double                 R0,
                                     double                 p0,
                                     double                 Ip,
                                     double                 beta,
                                     double                 B0,
                                     size_t                 n_axis_fix) noexcept
        {
            update_pq_rho_impl<SourceConstraintCode>(geometry, R0, p0, Ip, beta, B0, n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pq_psin(const GeometryRuntime& geometry,
                                      double                 R0,
                                      double                 p0,
                                      double                 Ip,
                                      double                 beta,
                                      double                 B0,
                                      size_t                 n_axis_fix) noexcept
        {
            update_pq_psin_impl<SourceConstraintCode>(geometry, R0, p0, Ip, beta, B0, n_axis_fix);
        }

        template <bool RhoCoordinate>
        constexpr void finalize_pressure_normalization(double p0, bool has_beta) noexcept
        {
            const RadialVector psin_r = const_root_row<root_psin_r>();
            double             numerator = 0.0;
            double             denominator = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double raw = materialized_pprime_input[i];
                denominator += raw * raw * GridType::weights[i];
                const double realized =
                    RhoCoordinate ? alpha1 * alpha2 * Pn_psin[i] * psin_r[i]
                                  : alpha1 * Pn_psin[i];
                numerator += realized * raw * GridType::weights[i];
            }

            double pressure_multiplier = 1.0;
            if (denominator > 1.0e-28)
                pressure_multiplier = numerator / denominator;
            else if (has_beta)
                pressure_multiplier = RhoCoordinate ? alpha1 * alpha2 : alpha1;

            RadialVector radial_pressure_gradient{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                radial_pressure_gradient[i] =
                    RhoCoordinate ? materialized_pprime_input[i]
                                  : alpha2 * materialized_pprime_input[i] * psin_r[i];
            }
            RadialVector pressure{uninitialized};
            compute_Pn_out(pressure, radial_pressure_gradient);
            double pressure_scale = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                pressure[i] = pressure_multiplier * (pressure[i] + p0);
                const double magnitude = math::abs(pressure[i]);
                if (magnitude > pressure_scale)
                    pressure_scale = magnitude;
            }

            const double normalized_alpha1 = pressure_scale / alpha2;
            const double source_rescale = alpha1 / normalized_alpha1;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                Pn_psin[i] *= source_rescale;
                FFn_psin[i] *= source_rescale;
            }
            alpha1 = normalized_alpha1;
        }

        template <size_t Row>
        constexpr RadialVector root_row() const noexcept
        {
            static_assert(Row < root_field_count, "source root row exceeds storage");

            RadialVector out{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                out[i] = source_target_root_fields(Row, i);
            return out;
        }

        template <size_t Row>
        constexpr void store_root_row(const RadialVector& values) noexcept
        {
            static_assert(Row < root_field_count, "source root row exceeds storage");

            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(Row, i) = values[i];
        }

    private:
        template <size_t Row>
        constexpr RadialVector const_root_row() const noexcept
        {
            return root_row<Row>();
        }

        template <size_t Row>
        constexpr RadialVector& root_row() noexcept = delete;

        constexpr void regularize_psin_r(size_t n_axis_fix) noexcept
        {
            if (n_axis_fix > 0 && n_axis_fix + 1 < radial_nodes)
            {
                const size_t anchor0  = n_axis_fix;
                const size_t anchor1  = n_axis_fix + 1;
                const double rho0     = GridType::nodes[anchor0];
                const double rho1     = GridType::nodes[anchor1];
                const double x0       = rho0 * rho0;
                const double x1       = rho1 * rho1;
                const double inv_rho0 = 1.0 / rho0;
                const double inv_rho1 = 1.0 / rho1;
                const double inv_dx   = 1.0 / (x1 - x0);
                const double slope0   = source_target_root_fields(root_psin_r, anchor0) * inv_rho0;
                const double slope1   = source_target_root_fields(root_psin_r, anchor1) * inv_rho1;
                const double gradient = (slope1 - slope0) * inv_dx;

                for (size_t i = 0; i < n_axis_fix; ++i)
                {
                    const double rho_i                        = GridType::nodes[i];
                    const double x_i                          = rho_i * rho_i;
                    source_target_root_fields(root_psin_r, i) = rho_i * (slope0 + gradient * (x_i - x0));
                }
            }

            for (size_t i = 0; i < radial_nodes; ++i)
                if (source_target_root_fields(root_psin_r, i) < 1.0e-10)
                    source_target_root_fields(root_psin_r, i) = 1.0e-10;
        }

        constexpr void update_psin_coordinate() noexcept
        {
            const auto   psin_r = const_root_row<root_psin_r>();
            RadialVector integrated{uninitialized};
            matvec_into(integrated, GridType::accumulator, psin_r);

            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);
        }

        constexpr void store_psin_coordinate(const RadialVector& integrated, double offset, double scale) noexcept
        {
            const double inv_scale = 1.0 / scale;
            if constexpr (radial_nodes == 1)
            {
                source_target_root_fields(root_psin, 0) = 1.0;
            }
            else
            {
                source_target_root_fields(root_psin, 0) = 0.0;
                for (size_t i = 1; i + 1 < radial_nodes; ++i)
                    source_target_root_fields(root_psin, i) = (integrated[i] - offset) * inv_scale;
                source_target_root_fields(root_psin, radial_nodes - 1) = 1.0;
            }
        }

        static constexpr void floor_signed_current_primitive(RadialVector& profile) noexcept
        {
            const double edge        = profile[radial_nodes - 1];
            const double abs_edge    = math::abs(edge);
            const double floor_value = (abs_edge > 1.0 ? abs_edge : 1.0) * 1.0e-12;
            if (edge < 0.0)
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    if (profile[i] > -floor_value)
                        profile[i] = -floor_value;
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    if (profile[i] < floor_value)
                        profile[i] = floor_value;
            }
        }

        static constexpr void regularize_axis_linear(RadialVector& profile, size_t n_axis_fix) noexcept
        {
            if (n_axis_fix == 0 || n_axis_fix + 1 >= radial_nodes)
                return;

            const size_t anchor0  = n_axis_fix;
            const size_t anchor1  = n_axis_fix + 1;
            const double rho0     = GridType::nodes[anchor0];
            const double rho1     = GridType::nodes[anchor1];
            const double x0       = rho0 * rho0;
            const double x1       = rho1 * rho1;
            const double inv_dx   = 1.0 / (x1 - x0);
            const double slope0   = profile[anchor0] / rho0;
            const double slope1   = profile[anchor1] / rho1;
            const double gradient = (slope1 - slope0) * inv_dx;

            for (size_t i = 0; i < n_axis_fix; ++i)
            {
                const double rho_i = GridType::nodes[i];
                const double x_i   = rho_i * rho_i;
                profile[i]         = rho_i * (slope0 + gradient * (x_i - x0));
            }
        }

        static constexpr void enforce_axis_even_profile(RadialVector& profile) noexcept
        {
            if constexpr (radial_nodes >= 3)
            {
                const double x1 = GridType::nodes[1] * GridType::nodes[1];
                const double x2 = GridType::nodes[2] * GridType::nodes[2];
                if (math::abs(x2 - x1) < 1.0e-14)
                    return;

                const double slope     = (profile[2] - profile[1]) / (x2 - x1);
                const double intercept = profile[1] - slope * x1;
                const double x0        = GridType::nodes[0] * GridType::nodes[0];
                profile[0]             = intercept + slope * x0;
                profile[1]             = intercept + slope * x1;
            }
        }

        static constexpr double clip_unit(double value) noexcept
        {
            if (value < 0.0)
                return 0.0;
            if (value > 1.0)
                return 1.0;
            return value;
        }

        static constexpr size_t local_uniform_stencil_start(double q) noexcept
        {
            if constexpr (stencil_size >= sample_count)
            {
                return 0;
            }
            else
            {
                const double pos    = q * static_cast<double>(sample_count - 1);
                size_t       center = static_cast<size_t>(pos);
                if (pos > static_cast<double>(center))
                    ++center;

                constexpr size_t half = stencil_size / 2;
                if (center < half)
                    return 0;

                const size_t     start     = center - half;
                constexpr size_t max_start = sample_count - stencil_size;
                return start > max_start ? max_start : start;
            }
        }

        constexpr void local_barycentric_interpolate_pair() noexcept
        {
            if constexpr (sample_count == 1)
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    materialized_pprime_input[i] = pprime_input[0];
                    materialized_driver_input[i] = driver_input[0];
                }
            }
            else
            {
                constexpr double denom_scale = static_cast<double>(sample_count - 1);
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    const double q         = clip_unit(source_parameter_query[i]);
                    const size_t start     = local_uniform_stencil_start(q);
                    const size_t nearest   = static_cast<size_t>(q * denom_scale + 0.5);
                    const double x_nearest = static_cast<double>(nearest) / denom_scale;
                    if (math::abs(q - x_nearest) <= 1.0e-14)
                    {
                        materialized_pprime_input[i] = pprime_input[nearest];
                        materialized_driver_input[i] = driver_input[nearest];
                        continue;
                    }

                    double denominator      = 0.0;
                    double numerator_pprime = 0.0;
                    double numerator_driver = 0.0;
                    for (size_t local_j = 0; local_j < stencil_size; ++local_j)
                    {
                        const size_t j = start + local_j;
                        const double term =
                            SourceShape::barycentric_weights[local_j] / (q - static_cast<double>(j) / denom_scale);
                        denominator += term;
                        numerator_pprime += term * pprime_input[j];
                        numerator_driver += term * driver_input[j];
                    }
                    materialized_pprime_input[i] = numerator_pprime / denominator;
                    materialized_driver_input[i] = numerator_driver / denominator;
                }
            }
        }

        constexpr void local_polynomial_interpolate_pair() noexcept
        {
            if constexpr (sample_count == 1)
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    materialized_pprime_input[i] = pprime_input[0];
                    materialized_driver_input[i] = driver_input[0];
                }
            }
            else
            {
                constexpr double interval_count = static_cast<double>(sample_count - 1);
                constexpr size_t last_interval  = sample_count - 2;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    const double q = clip_unit(source_parameter_query[i]);
                    double       t = 0.0;
                    size_t       interval = static_cast<size_t>(q * interval_count);
                    if (interval > last_interval)
                    {
                        interval = last_interval;
                        t        = 1.0;
                    }
                    else
                    {
                        t = q * interval_count - static_cast<double>(interval);
                    }
                    materialized_pprime_input[i] =
                        evaluate_local_polynomial_value(pprime_input, interval, t);
                    materialized_driver_input[i] =
                        evaluate_local_polynomial_value(driver_input, interval, t);
                }
            }
        }

        static constexpr double evaluate_local_polynomial_value(const SourceVector& samples,
                                                                size_t              interval,
                                                                double              t) noexcept
        {
            if constexpr (sample_count == 1)
            {
                (void)interval;
                (void)t;
                return samples[0];
            }
            else if constexpr (sample_count == 2)
            {
                (void)interval;
                const double y0 = samples[0];
                const double y1 = samples[1];
                return y0 + t * (y1 - y0);
            }
            else if constexpr (sample_count == 3)
            {
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;
                if (interval == 0)
                {
                    const double y0 = samples[0];
                    const double y1 = samples[1];
                    const double y2 = samples[2];
                    c0              = y0;
                    c1              = -1.5 * y0 + 2.0 * y1 - 0.5 * y2;
                    c2              = 0.5 * y0 - y1 + 0.5 * y2;
                }
                else
                {
                    const double y0 = samples[interval - 1];
                    const double y1 = samples[interval];
                    const double y2 = samples[interval + 1];
                    c0              = y1;
                    c1              = -0.5 * y0 + 0.5 * y2;
                    c2              = 0.5 * y0 - y1 + 0.5 * y2;
                }
                return (c2 * t + c1) * t + c0;
            }
            else
            {
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;
                double c3 = 0.0;
                if (interval == 0)
                {
                    const double y0 = samples[0];
                    const double y1 = samples[1];
                    const double y2 = samples[2];
                    const double y3 = samples[3];
                    c0              = y0;
                    c1              = (-11.0 * y0 + 18.0 * y1 - 9.0 * y2 + 2.0 * y3) / 6.0;
                    c2              = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                    c3              = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0;
                }
                else if (interval == sample_count - 2)
                {
                    const double y0 = samples[interval - 2];
                    const double y1 = samples[interval - 1];
                    const double y2 = samples[interval];
                    const double y3 = samples[interval + 1];
                    c0              = y2;
                    c1              = (y0 - 6.0 * y1 + 3.0 * y2 + 2.0 * y3) / 6.0;
                    c2              = 0.5 * y1 - y2 + 0.5 * y3;
                    c3              = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0;
                }
                else
                {
                    const double y0 = samples[interval - 1];
                    const double y1 = samples[interval];
                    const double y2 = samples[interval + 1];
                    const double y3 = samples[interval + 2];
                    c0              = y1;
                    c1              = (-2.0 * y0 - 3.0 * y1 + 6.0 * y2 - y3) / 6.0;
                    c2              = 0.5 * y0 - y1 + 0.5 * y2;
                    c3              = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0;
                }
                return ((c3 * t + c2) * t + c1) * t + c0;
            }
        }

        constexpr void seed_psin_query_from_passive_psin_profile() noexcept
        {
            if constexpr (radial_nodes == 1)
            {
                source_psin_query[0] = 1.0;
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    source_psin_query[i] = GridType::nodes[i] * GridType::nodes[i];
                source_psin_query[0] = 0.0;
                source_psin_query[radial_nodes - 1] = 1.0;
            }
        }

        constexpr bool update_fixed_point_psin_query() noexcept
        {
            double max_abs_diff = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double psin = source_target_root_fields(root_psin, i);
                const double diff = math::abs(psin - source_psin_query[i]);
                if (diff > max_abs_diff)
                    max_abs_diff = diff;
                source_psin_query[i] = psin;
            }
            return max_abs_diff <= pj2_psin_uniform_fixed_point_max_residual;
        }

        constexpr void copy_source_target_to_profile_root() noexcept
        {
            for (size_t row = 0; row < root_field_count; ++row)
                for (size_t i = 0; i < radial_nodes; ++i)
                    profile_root_fields(row, i) = source_target_root_fields(row, i);
        }

        template <typename GeometryRuntime>
        constexpr void fill_pf_psin_integrand(RadialVector& out, const GeometryRuntime& geometry) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                out[i] = materialized_driver_input[i] * radial_Ln_r[i] +
                         radial_V_r[i] * materialized_pprime_input[i] * pressure_factor;
            }
        }

        template <typename GeometryRuntime>
        constexpr void fill_pf_rho_integrand(RadialVector& out, const GeometryRuntime& geometry) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Kn = geometry_radial + geometry::radial_Kn * radial_nodes;
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                out[i] = radial_Kn[i] *
                         (materialized_driver_input[i] * radial_Ln_r[i] +
                         radial_V_r[i] * materialized_pprime_input[i] * pressure_factor);
            }
        }

        template <typename GeometryRuntime>
        constexpr void fill_pp_ffn_psin(RadialVector&          out,
                                        const RadialVector&    psin_r,
                                        const RadialVector&    psin_rr,
                                        const GeometryRuntime& geometry,
                                        double                 alpha_ratio) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Kn = geometry_radial + geometry::radial_Kn * radial_nodes;
            const double* const radial_Kn_r = geometry_radial + geometry::radial_Kn_r * radial_nodes;
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double term0 = alpha_ratio * (radial_Kn_r[i] * psin_r[i] + radial_Kn[i] * psin_rr[i]);
                const double term1 = radial_V_r[i] * Pn_psin[i] * pressure_factor;
                out[i]             = -(term0 + term1) / radial_Ln_r[i];
            }
        }

        template <typename GeometryRuntime>
        constexpr void fill_pi_ffn_psin(RadialVector&          out,
                                        const RadialVector&    Itor_r,
                                        const GeometryRuntime& geometry,
                                        double                 current_scale) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double term0 = current_scale * Itor_r[i];
                const double term1 = radial_V_r[i] * Pn_psin[i] * pressure_factor;
                out[i]             = -(term0 + term1) / radial_Ln_r[i];
            }
        }

        template <typename GeometryRuntime>
        constexpr void fill_pj_ffn_psin(RadialVector&          out,
                                        const RadialVector&    jtor,
                                        const GeometryRuntime& geometry,
                                        double                 current_scale) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_S_r = geometry_radial + geometry::radial_S_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double term0 = current_scale * jtor[i] * radial_S_r[i];
                const double term1 = radial_V_r[i] * Pn_psin[i] * pressure_factor;
                out[i]             = -(term0 + term1) / radial_Ln_r[i];
            }
        }

        template <typename Weights>
        static constexpr double dot(const RadialVector& values, const Weights& weights) noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += values[i] * weights[i];
            return total;
        }

        template <typename GeometryRuntime>
        constexpr double g1n_psin_integral_from_radial_moments(const GeometryRuntime& geometry) const noexcept
        {
            constexpr double two_pi     = 2.0 * geometry::detail::pi;
            constexpr double inv_two_pi = 1.0 / two_pi;
            double           total      = 0.0;
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                total += GridType::weights[i] *
                         (two_pi * radial_Ln_r[i] * FFn_psin[i] + inv_two_pi * radial_V_r[i] * Pn_psin[i]);
            }
            return total;
        }

        template <typename GeometryRuntime>
        constexpr double
        g1n_rho_integral_from_radial_moments(const GeometryRuntime& geometry, double source_scale) const noexcept
        {
            constexpr double two_pi     = 2.0 * geometry::detail::pi;
            constexpr double inv_two_pi = 1.0 / two_pi;
            const RadialVector psin_r   = const_root_row<root_psin_r>();
            double             total    = 0.0;
            const double* const geometry_radial = geometry.radial_fields.aligned_data();
            const double* const radial_Ln_r = geometry_radial + geometry::radial_Ln_r * radial_nodes;
            const double* const radial_V_r = geometry_radial + geometry::radial_V_r * radial_nodes;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                total += GridType::weights[i] * source_scale / psin_r[i] *
                         (two_pi * radial_Ln_r[i] * materialized_driver_input[i] +
                          inv_two_pi * radial_V_r[i] * materialized_pprime_input[i]);
            }
            return total;
        }

        constexpr void compute_Pn_out(RadialVector& out, const RadialVector& Pn_r) const noexcept
        {
            radial_grid_accumulator_matvec_into<GridType>(out, Pn_r);
            const double offset = dot(Pn_r, GridType::weights);
            for (size_t i = 0; i < radial_nodes; ++i)
                out[i] -= offset;
        }

        template <bool RhoCoordinate>
        constexpr double ensure_pressure_alpha1(double              candidate,
                                                double              p0,
                                                double              flux_scale,
                                                const RadialVector& psin_r) const noexcept
        {
            if (math::is_finite(candidate) && math::abs(candidate) > 1.0e-14)
                return candidate;
            RadialVector radial_pressure_gradient{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                radial_pressure_gradient[i] =
                    RhoCoordinate ? materialized_pprime_input[i]
                                  : flux_scale * materialized_pprime_input[i] * psin_r[i];
            }
            RadialVector pressure{uninitialized};
            compute_Pn_out(pressure, radial_pressure_gradient);
            double pressure_scale = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double magnitude = math::abs(pressure[i] + p0);
                if (magnitude > pressure_scale)
                    pressure_scale = magnitude;
            }
            return pressure_scale / flux_scale;
        }

        template <int SourceConstraintCode, bool RhoCoordinate, typename GeometryRuntime>
        constexpr void update_pp(const GeometryRuntime& geometry,
                                 double                 p0,
                                 double                 Ip,
                                 double                 beta,
                                 double                 B0,
                                 size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PP source supports null, Ip, beta, or Ip_beta constraints");

            RadialVector psin_r{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] = materialized_driver_input[i];
                alpha2 = Ip /
                         (2.0 * geometry::detail::pi *
                          geometry.radial_field(geometry::radial_Kn, radial_nodes - 1) * psin_r[radial_nodes - 1]);
            }
            else
            {
                alpha2 = dot(materialized_driver_input, GridType::weights);
                const double inv_alpha2 = 1.0 / alpha2;
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] = materialized_driver_input[i] * inv_alpha2;
                (void)Ip;
            }

            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    if constexpr (RhoCoordinate)
                    {
                        Pn_psin[i] = materialized_pprime_input[i] / psin_r[i];
                        Pn_r[i]    = materialized_pprime_input[i];
                    }
                    else
                    {
                        Pn_psin[i] = materialized_pprime_input[i];
                        Pn_r[i]    = materialized_pprime_input[i] * psin_r[i];
                    }
                }
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, Pn_r);
                const double volume =
                    dot_radial_moment(geometry, geometry::radial_V_r);
                const double relative =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r);
                const double pressure_denominator =
                    RhoCoordinate ? alpha2 * (relative + p0 * volume)
                                  : alpha2 * relative + p0 * volume;
                alpha1 = 0.5 * beta * B0 * B0 * volume /
                         pressure_denominator;
            }
            else
            {
                if constexpr (RhoCoordinate)
                {
                    alpha1               = -dot(materialized_pprime_input, GridType::weights) / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double scaling = 1.0 / (alpha1 * alpha2);
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * scaling / psin_r[i];
                }
                else
                {
                    double pressure_total = 0.0;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        pressure_total += materialized_pprime_input[i] * psin_r[i] * alpha2 * GridType::weights[i];
                    alpha1 = -pressure_total / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double inv_alpha1 = 1.0 / alpha1;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * inv_alpha1;
                }
                (void)beta;
                (void)B0;
            }

            fill_pp_ffn_psin(FFn_psin, psin_r, psin_rr, geometry, alpha2 / alpha1);
            regularize_ffn_psin(n_axis_fix);
        }

        template <int SourceConstraintCode, bool RhoCoordinate, typename GeometryRuntime>
        constexpr void update_pi(const GeometryRuntime& geometry,
                                 double                 p0,
                                 double                 Ip,
                                 double                 beta,
                                 double                 B0,
                                 size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PI source supports null, Ip, beta, or Ip_beta constraints");

            RadialVector Itor{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                const double source_scale = Ip / materialized_driver_input[radial_nodes - 1];
                for (size_t i = 0; i < radial_nodes; ++i)
                    Itor[i] = materialized_driver_input[i] * source_scale;
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    Itor[i] = materialized_driver_input[i];
                (void)Ip;
            }
            floor_signed_current_primitive(Itor);

            constexpr double inv_two_pi = 1.0 / (2.0 * geometry::detail::pi);
            RadialVector     itor_over_kn{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                itor_over_kn[i] = Itor[i] / geometry.radial_field(geometry::radial_Kn, i) * inv_two_pi;
            alpha2 = dot(itor_over_kn, GridType::weights);

            RadialVector psin_r{uninitialized};
            const double psin_scale = inv_two_pi / alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] = Itor[i] / geometry.radial_field(geometry::radial_Kn, i) * psin_scale;
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            RadialVector Itor_r{uninitialized};
            matvec_into(Itor_r, GridType::differentiator, Itor);
            regularize_axis_linear(Itor_r, n_axis_fix);

            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    if constexpr (RhoCoordinate)
                    {
                        Pn_psin[i] = materialized_pprime_input[i] / psin_r[i];
                        Pn_r[i]    = materialized_pprime_input[i];
                    }
                    else
                    {
                        Pn_psin[i] = materialized_pprime_input[i];
                        Pn_r[i]    = materialized_pprime_input[i] * psin_r[i];
                    }
                }
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, Pn_r);
                const double volume =
                    dot_radial_moment(geometry, geometry::radial_V_r);
                const double relative =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r);
                const double pressure_denominator =
                    RhoCoordinate ? alpha2 * (relative + p0 * volume)
                                  : alpha2 * relative + p0 * volume;
                alpha1 = 0.5 * beta * B0 * B0 * volume /
                         pressure_denominator;
            }
            else
            {
                if constexpr (RhoCoordinate)
                {
                    alpha1               = -dot(materialized_pprime_input, GridType::weights) / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double scaling = 1.0 / (alpha1 * alpha2);
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * scaling / psin_r[i];
                }
                else
                {
                    double pressure_total = 0.0;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        pressure_total += materialized_pprime_input[i] * psin_r[i] * alpha2 * GridType::weights[i];
                    alpha1 = -pressure_total / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double inv_alpha1 = 1.0 / alpha1;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * inv_alpha1;
                }
                (void)beta;
                (void)B0;
            }

            fill_pi_ffn_psin(FFn_psin, Itor_r, geometry, inv_two_pi / alpha1);
            regularize_ffn_psin(n_axis_fix);
        }

        template <int SourceConstraintCode, bool RhoCoordinate, typename GeometryRuntime>
        constexpr void update_pj1(const GeometryRuntime& geometry,
                                  double                 p0,
                                  double                 Ip,
                                  double                 beta,
                                  double                 B0,
                                  size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PJ1 source supports null, Ip, beta, or Ip_beta constraints");

            RadialVector integrand_j{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                integrand_j[i] = materialized_driver_input[i] * geometry.radial_field(geometry::radial_S_r, i);

            RadialVector I_tor_prof{uninitialized};
            radial_grid_accumulator_matvec_into<GridType>(I_tor_prof, integrand_j);

            RadialVector I_tor{uninitialized};
            RadialVector jtor{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                const double source_scale = Ip / I_tor_prof[radial_nodes - 1];
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    I_tor[i] = I_tor_prof[i] * source_scale;
                    jtor[i]  = materialized_driver_input[i] * source_scale;
                }
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    I_tor[i] = I_tor_prof[i];
                    jtor[i]  = materialized_driver_input[i];
                }
                (void)Ip;
            }

            enforce_axis_even_profile(jtor);
            floor_signed_current_primitive(I_tor);

            constexpr double inv_two_pi = 1.0 / (2.0 * geometry::detail::pi);
            RadialVector     itor_over_kn{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                itor_over_kn[i] = I_tor[i] / geometry.radial_field(geometry::radial_Kn, i) * inv_two_pi;
            alpha2 = dot(itor_over_kn, GridType::weights);

            RadialVector psin_r{uninitialized};
            const double psin_scale = inv_two_pi / alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] = I_tor[i] / geometry.radial_field(geometry::radial_Kn, i) * psin_scale;
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    if constexpr (RhoCoordinate)
                    {
                        Pn_psin[i] = materialized_pprime_input[i] / psin_r[i];
                        Pn_r[i]    = materialized_pprime_input[i];
                    }
                    else
                    {
                        Pn_psin[i] = materialized_pprime_input[i];
                        Pn_r[i]    = materialized_pprime_input[i] * psin_r[i];
                    }
                }
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, Pn_r);
                const double volume =
                    dot_radial_moment(geometry, geometry::radial_V_r);
                const double relative =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r);
                const double pressure_denominator =
                    RhoCoordinate ? alpha2 * (relative + p0 * volume)
                                  : alpha2 * relative + p0 * volume;
                alpha1 = 0.5 * beta * B0 * B0 * volume /
                         pressure_denominator;
            }
            else
            {
                if constexpr (RhoCoordinate)
                {
                    alpha1               = -dot(materialized_pprime_input, GridType::weights) / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double scaling = 1.0 / (alpha1 * alpha2);
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * scaling / psin_r[i];
                }
                else
                {
                    double pressure_total = 0.0;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        pressure_total += materialized_pprime_input[i] * psin_r[i] * alpha2 * GridType::weights[i];
                    alpha1 = -pressure_total / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double inv_alpha1 = 1.0 / alpha1;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * inv_alpha1;
                }
                (void)beta;
                (void)B0;
            }

            fill_pj_ffn_psin(FFn_psin, jtor, geometry, inv_two_pi / alpha1);
            regularize_ffn_psin(n_axis_fix);
        }

        template <int SourceConstraintCode, bool RhoCoordinate, typename GeometryRuntime>
        constexpr void update_pj2(const GeometryRuntime& geometry,
                                  double                 R0,
                                  double                 p0,
                                  double                 Ip,
                                  double                 beta,
                                  double                 B0,
                                  size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PJ2 source supports null, Ip, beta, or Ip_beta constraints");
            (void)R0;
            (void)B0;

            RadialVector integrand{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                integrand[i] = geometry.radial_field(geometry::radial_Ln_r, i) *
                               materialized_driver_input[i] / active_F[i];
            }

            RadialVector integral_val{uninitialized};
            radial_grid_accumulator_matvec_into<GridType>(integral_val, integrand);

            RadialVector I_tor{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                const double edge_F       = RhoCoordinate ? active_F[radial_nodes - 1] : R0 * B0;
                const double source_scale = Ip / (edge_F * integral_val[radial_nodes - 1]);
                for (size_t i = 0; i < radial_nodes; ++i)
                    I_tor[i] = active_F[i] * integral_val[i] * source_scale;
            }
            else
            {
                constexpr double two_pi = 2.0 * geometry::detail::pi;
                for (size_t i = 0; i < radial_nodes; ++i)
                    I_tor[i] = active_F[i] * integral_val[i] * two_pi;
                (void)Ip;
            }

            constexpr double inv_two_pi = 1.0 / (2.0 * geometry::detail::pi);
            RadialVector     itor_over_kn{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                itor_over_kn[i] = I_tor[i] / geometry.radial_field(geometry::radial_Kn, i) * inv_two_pi;
            alpha2 = dot(itor_over_kn, GridType::weights);

            RadialVector psin_r{uninitialized};
            const double inv_alpha2 = 1.0 / alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] = itor_over_kn[i] * inv_alpha2;
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    if constexpr (RhoCoordinate)
                    {
                        Pn_psin[i] = materialized_pprime_input[i] / psin_r[i];
                        Pn_r[i]    = materialized_pprime_input[i];
                    }
                    else
                    {
                        Pn_psin[i] = materialized_pprime_input[i];
                        Pn_r[i]    = materialized_pprime_input[i] * psin_r[i];
                    }
                }
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, Pn_r);
                const double volume =
                    dot_radial_moment(geometry, geometry::radial_V_r);
                const double relative =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r);
                const double pressure_denominator =
                    RhoCoordinate ? alpha2 * (relative + p0 * volume)
                                  : alpha2 * relative + p0 * volume;
                alpha1 = 0.5 * beta * B0 * B0 * volume /
                         pressure_denominator;
            }
            else
            {
                if constexpr (RhoCoordinate)
                {
                    alpha1               = -dot(materialized_pprime_input, GridType::weights) / alpha2;
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double scaling = 1.0 / (alpha1 * alpha2);
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * scaling / psin_r[i];
                }
                else
                {
                    alpha1 = -weighted_dot(materialized_pprime_input, psin_r, GridType::weights);
                    alpha1 = ensure_pressure_alpha1<RhoCoordinate>(
                        alpha1, p0, alpha2, psin_r);
                    const double inv_alpha1 = 1.0 / alpha1;
                    for (size_t i = 0; i < radial_nodes; ++i)
                        Pn_psin[i] = materialized_pprime_input[i] * inv_alpha1;
                }
                (void)beta;
            }

            const double ffn_scale = 1.0 / (alpha1 * alpha2);
            for (size_t i = 0; i < radial_nodes; ++i)
                FFn_psin[i] = active_F[i] * active_F_r[i] * ffn_scale / psin_r[i];
            regularize_ffn_psin(n_axis_fix);
        }

        template <typename Differentiator>
        static constexpr void solve_pq_linear_system(RadialVector&          solution,
                                                     const Differentiator&  differentiator,
                                                     const RadialVector&    coeff_d,
                                                     const RadialVector&    coeff_y,
                                                     const RadialVector&    forcing,
                                                     double                 edge_value) noexcept
        {
            Matrix<double, radial_nodes, radial_nodes> matrix{uninitialized};
            Matrix<double, radial_nodes, 1>            rhs{uninitialized};

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                for (size_t j = 0; j < radial_nodes; ++j)
                    matrix(i, j) = coeff_d[i] * differentiator(i, j);
                matrix(i, i) += coeff_y[i];
                rhs(i, 0) = forcing[i];
            }

            constexpr size_t edge = radial_nodes - 1;
            for (size_t j = 0; j < radial_nodes; ++j)
                matrix(edge, j) = 0.0;
            matrix(edge, edge) = 1.0;
            rhs(edge, 0)       = edge_value;

            const auto solved = linalg::solve<linalg::Doolittle>(matrix, rhs);
            for (size_t i = 0; i < radial_nodes; ++i)
                solution[i] = solved(i, 0);
        }

        template <typename Differentiator>
        static constexpr void solve_pq_linear_system_two_rhs(RadialVector&          solution0,
                                                             RadialVector&          solution1,
                                                             const Differentiator&  differentiator,
                                                             const RadialVector&    coeff_d,
                                                             const RadialVector&    coeff_y,
                                                             const RadialVector&    forcing0,
                                                             const RadialVector&    forcing1,
                                                             double                 edge_value0,
                                                             double                 edge_value1) noexcept
        {
            Matrix<double, radial_nodes, radial_nodes> matrix{uninitialized};
            Matrix<double, radial_nodes, 2>            rhs{uninitialized};

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                for (size_t j = 0; j < radial_nodes; ++j)
                    matrix(i, j) = coeff_d[i] * differentiator(i, j);
                matrix(i, i) += coeff_y[i];
                rhs(i, 0) = forcing0[i];
                rhs(i, 1) = forcing1[i];
            }

            constexpr size_t edge = radial_nodes - 1;
            for (size_t j = 0; j < radial_nodes; ++j)
                matrix(edge, j) = 0.0;
            matrix(edge, edge) = 1.0;
            rhs(edge, 0)       = edge_value0;
            rhs(edge, 1)       = edge_value1;

            const auto solved = linalg::solve<linalg::Doolittle>(matrix, rhs);
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                solution0[i] = solved(i, 0);
                solution1[i] = solved(i, 1);
            }
        }

        template <typename GeometryRuntime>
        constexpr double pq_psin_beta_residual(const GeometryRuntime& geometry,
                                               double                 trial_alpha1,
                                               const RadialVector&    F0,
                                               const RadialVector&    F1,
                                               const RadialVector&    q_prof,
                                               RadialVector&          trial_psin_r,
                                               RadialVector&          trial_Pn_r,
                                               RadialVector&          trial_Pn,
                                               double                 p0,
                                               double                 beta_target) const noexcept
        {
            double trial_alpha2 = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double F_value = F0[i] + trial_alpha1 * F1[i];
                const double psi_r =
                    F_value * geometry.radial_field(geometry::radial_Ln_r, i) / q_prof[i];
                if (!math::is_finite(psi_r))
                    return std::numeric_limits<double>::quiet_NaN();
                trial_psin_r[i] = psi_r;
                trial_alpha2 += psi_r * GridType::weights[i];
            }
            if (!math::is_finite(trial_alpha2) || math::abs(trial_alpha2) <= 1.0e-14)
                return std::numeric_limits<double>::quiet_NaN();
            const double inv_alpha2 = 1.0 / trial_alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                trial_psin_r[i] *= inv_alpha2;
                if (!math::is_finite(trial_psin_r[i]) || trial_psin_r[i] <= 0.0)
                    return std::numeric_limits<double>::quiet_NaN();
                trial_Pn_r[i] = materialized_pprime_input[i] * trial_psin_r[i];
            }
            compute_Pn_out(trial_Pn, trial_Pn_r);
            const double beta_den = weighted_dot(trial_Pn, geometry, geometry::radial_V_r);
            if (!math::is_finite(beta_den))
                return std::numeric_limits<double>::quiet_NaN();
            return trial_alpha1 *
                       (p0 * dot_radial_moment(geometry, geometry::radial_V_r) +
                        trial_alpha2 * beta_den) -
                   beta_target;
        }

        template <typename GeometryRuntime>
        constexpr double solve_pq_psin_beta_alpha1(const GeometryRuntime& geometry,
                                                   const RadialVector&    F0,
                                                   const RadialVector&    F1,
                                                   const RadialVector&    q_prof,
                                                   double                 p0,
                                                   double                 beta_target) const noexcept
        {
            RadialVector trial_psin_r{uninitialized};
            RadialVector trial_Pn_r{uninitialized};
            RadialVector trial_Pn{uninitialized};

            constexpr double base = 0.0;
            const double     r_base =
                pq_psin_beta_residual(
                    geometry, base, F0, F1, q_prof, trial_psin_r, trial_Pn_r, trial_Pn, p0, beta_target);

            for (size_t direction_index = 0; direction_index < 2; ++direction_index)
            {
                const double direction = direction_index == 0 ? 1.0 : -1.0;
                double upper  = direction;
                double r_upper = pq_psin_beta_residual(
                    geometry, upper, F0, F1, q_prof, trial_psin_r, trial_Pn_r, trial_Pn, p0, beta_target);
                for (size_t bracket_iter = 0; bracket_iter < 80; ++bracket_iter)
                {
                    if (math::is_finite(r_base) && math::is_finite(r_upper) && r_base * r_upper <= 0.0)
                    {
                        double lower   = base;
                        double r_lower = r_base;
                        for (size_t iter = 0; iter < 80; ++iter)
                        {
                            const double mid = 0.5 * (lower + upper);
                            const double r_mid = pq_psin_beta_residual(
                                geometry, mid, F0, F1, q_prof, trial_psin_r, trial_Pn_r, trial_Pn, p0, beta_target);
                            if (!math::is_finite(r_mid))
                            {
                                upper = mid;
                                continue;
                            }
                            if (math::abs(r_mid) <= 1.0e-12 * (1.0 + math::abs(beta_target)))
                                return mid;
                            if (r_lower * r_mid <= 0.0)
                            {
                                upper  = mid;
                                r_upper = r_mid;
                            }
                            else
                            {
                                lower   = mid;
                                r_lower = r_mid;
                            }
                        }
                        return 0.5 * (lower + upper);
                    }
                    upper *= 2.0;
                    r_upper = pq_psin_beta_residual(
                        geometry, upper, F0, F1, q_prof, trial_psin_r, trial_Pn_r, trial_Pn, p0, beta_target);
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pq_rho_impl(const GeometryRuntime& geometry,
                                          double                 R0,
                                          double                 p0,
                                          double                 Ip,
                                          double                 beta,
                                          double                 B0,
                                          size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PQ/rho source supports null, Ip, beta, or Ip_beta constraints");

            const double edge_F = R0 * B0;

            RadialVector q_prof{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                constexpr double two_pi = 2.0 * geometry::detail::pi;
                const double     edge_scale =
                    two_pi * edge_F / Ip *
                    geometry.radial_field(geometry::radial_Kn, radial_nodes - 1) *
                    geometry.radial_field(geometry::radial_Ln_r, radial_nodes - 1) /
                    materialized_driver_input[radial_nodes - 1];
                for (size_t i = 0; i < radial_nodes; ++i)
                    q_prof[i] = materialized_driver_input[i] * edge_scale;
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    q_prof[i] = materialized_driver_input[i];
                (void)Ip;
            }

            RadialVector W{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                W[i] = geometry.radial_field(geometry::radial_Kn, i) *
                       geometry.radial_field(geometry::radial_Ln_r, i) / q_prof[i];
            }

            RadialVector W_r{uninitialized};
            matvec_into(W_r, GridType::differentiator, W);

            double pressure_scale = 1.0;
            double beta_C         = 0.0;
            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, materialized_pprime_input);
                const double beta_den_pre =
                    weighted_dot(Pn_out, geometry, geometry::radial_V_r) +
                    p0 * dot_radial_moment(geometry, geometry::radial_V_r);
                beta_C = 0.5 * beta * B0 * B0 *
                         dot_radial_moment(geometry, geometry::radial_V_r) / beta_den_pre;
                pressure_scale = beta_C;
            }
            else
            {
                (void)beta;
            }

            constexpr double pressure_factor =
                1.0 / (2.0 * geometry::detail::pi * geometry::detail::pi);
            RadialVector coeff_d{uninitialized};
            RadialVector coeff_y{uninitialized};
            RadialVector rhs{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                coeff_d[i] = W[i] + q_prof[i];
                coeff_y[i] = 2.0 * W_r[i];
                rhs[i] = -pressure_factor * pressure_scale *
                         geometry.radial_field(geometry::radial_V_r, i) *
                         materialized_pprime_input[i] * q_prof[i] /
                         geometry.radial_field(geometry::radial_Ln_r, i);
            }

            RadialVector Y{uninitialized};
            solve_pq_linear_system(
                Y, GridType::differentiator, coeff_d, coeff_y, rhs, edge_F * edge_F);

            const double sign_F = edge_F < 0.0 ? -1.0 : 1.0;
            RadialVector psin_r{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double F_i = sign_F * math::sqrt(Y[i]);
                psin_r[i] = F_i * geometry.radial_field(geometry::radial_Ln_r, i) / q_prof[i];
            }

            alpha2 = dot(psin_r, GridType::weights);
            const double inv_alpha2 = 1.0 / alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] *= inv_alpha2;
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                alpha1 = beta_C / alpha2;
                for (size_t i = 0; i < radial_nodes; ++i)
                    Pn_psin[i] = materialized_pprime_input[i] / psin_r[i];
            }
            else
            {
                alpha1               = -dot(materialized_pprime_input, GridType::weights) / alpha2;
                alpha1 = ensure_pressure_alpha1<true>(
                    alpha1, p0, alpha2, psin_r);
                const double scaling = 1.0 / (alpha1 * alpha2);
                for (size_t i = 0; i < radial_nodes; ++i)
                    Pn_psin[i] = materialized_pprime_input[i] * scaling / psin_r[i];
            }

            RadialVector Y_r{uninitialized};
            matvec_into(Y_r, GridType::differentiator, Y);
            const double ffn_scale = 0.5 / (alpha1 * alpha2);
            for (size_t i = 0; i < radial_nodes; ++i)
                FFn_psin[i] = Y_r[i] * ffn_scale / psin_r[i];
            regularize_ffn_psin(n_axis_fix);
        }

        template <int SourceConstraintCode, typename GeometryRuntime>
        constexpr void update_pq_psin_impl(const GeometryRuntime& geometry,
                                           double                 R0,
                                           double                 p0,
                                           double                 Ip,
                                           double                 beta,
                                           double                 B0,
                                           size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");
            static_assert(SourceConstraintCode == 0 || SourceConstraintCode == 1 || SourceConstraintCode == 2 ||
                              SourceConstraintCode == 3,
                          "PQ/psin source supports null, Ip, beta, or Ip_beta constraints");

            const double edge_F = R0 * B0;

            RadialVector q_prof{uninitialized};
            if constexpr (SourceConstraintCode == 1 || SourceConstraintCode == 3)
            {
                constexpr double two_pi = 2.0 * geometry::detail::pi;
                const double     edge_scale =
                    two_pi * edge_F / Ip *
                    geometry.radial_field(geometry::radial_Kn, radial_nodes - 1) *
                    geometry.radial_field(geometry::radial_Ln_r, radial_nodes - 1) /
                    materialized_driver_input[radial_nodes - 1];
                for (size_t i = 0; i < radial_nodes; ++i)
                    q_prof[i] = materialized_driver_input[i] * edge_scale;
            }
            else
            {
                for (size_t i = 0; i < radial_nodes; ++i)
                    q_prof[i] = materialized_driver_input[i];
                (void)Ip;
            }

            RadialVector W{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                W[i] = geometry.radial_field(geometry::radial_Kn, i) *
                       geometry.radial_field(geometry::radial_Ln_r, i) / q_prof[i];
            }

            RadialVector W_r{uninitialized};
            matvec_into(W_r, GridType::differentiator, W);

            RadialVector coeff_d{uninitialized};
            RadialVector coeff_y{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                coeff_d[i] = W[i] + q_prof[i];
                coeff_y[i] = W_r[i];
            }

            constexpr double pressure_factor =
                1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            RadialVector F_solved{uninitialized};
            if constexpr (SourceConstraintCode == 2 || SourceConstraintCode == 3)
            {
                RadialVector F0_rhs{uninitialized};
                RadialVector F1_rhs{uninitialized};
                RadialVector F1{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    F0_rhs[i] = 0.0;
                    F1_rhs[i] = -pressure_factor *
                                geometry.radial_field(geometry::radial_V_r, i) *
                                materialized_pprime_input[i];
                }
                solve_pq_linear_system_two_rhs(
                    F_solved,
                    F1,
                    GridType::differentiator,
                    coeff_d,
                    coeff_y,
                    F0_rhs,
                    F1_rhs,
                    edge_F,
                    0.0);

                const double beta_target =
                    0.5 * beta * B0 * B0 * dot_radial_moment(geometry, geometry::radial_V_r);
                alpha1 = solve_pq_psin_beta_alpha1(
                    geometry, F_solved, F1, q_prof, p0, beta_target);
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    F_solved[i] += alpha1 * F1[i];
                    Pn_psin[i] = materialized_pprime_input[i];
                }
            }
            else
            {
                RadialVector rhs{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    rhs[i] = -pressure_factor *
                             geometry.radial_field(geometry::radial_V_r, i) *
                             materialized_pprime_input[i];
                }
                solve_pq_linear_system(
                    F_solved, GridType::differentiator, coeff_d, coeff_y, rhs, edge_F);
                (void)beta;
            }

            RadialVector psin_r{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                psin_r[i] = F_solved[i] * geometry.radial_field(geometry::radial_Ln_r, i) / q_prof[i];
            }

            alpha2 = dot(psin_r, GridType::weights);
            const double inv_alpha2 = 1.0 / alpha2;
            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] *= inv_alpha2;
            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            RadialVector psin_rr{uninitialized};
            RadialVector integrated{uninitialized};
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            store_psin_coordinate(integrated, offset, scale);

            if constexpr (SourceConstraintCode == 0 || SourceConstraintCode == 1)
            {
                alpha1 = -weighted_dot(materialized_pprime_input, psin_r, GridType::weights);
                alpha1 = ensure_pressure_alpha1<false>(
                    alpha1, p0, alpha2, psin_r);
                const double inv_alpha1 = 1.0 / alpha1;
                for (size_t i = 0; i < radial_nodes; ++i)
                    Pn_psin[i] = materialized_pprime_input[i] * inv_alpha1;
            }

            RadialVector F_r{uninitialized};
            matvec_into(F_r, GridType::differentiator, F_solved);
            const double inv_alpha1 = 1.0 / alpha1;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                FFn_psin[i] = q_prof[i] * F_r[i] /
                              geometry.radial_field(geometry::radial_Ln_r, i) * inv_alpha1;
            }
            regularize_ffn_psin(n_axis_fix);
        }

        template <typename GeometryRuntime>
        constexpr double dot_radial_moment(const GeometryRuntime& geometry, size_t row) const noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += geometry.radial_field(row, i) * GridType::weights[i];
            return total;
        }

        template <typename GeometryRuntime>
        constexpr double weighted_dot(const RadialVector& values,
                                      const GeometryRuntime& geometry,
                                      size_t row) const noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += values[i] * geometry.radial_field(row, i) * GridType::weights[i];
            return total;
        }

        template <typename Weights>
        static constexpr double weighted_dot(const RadialVector& lhs,
                                             const RadialVector& rhs,
                                             const Weights&      weights) noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += lhs[i] * rhs[i] * weights[i];
            return total;
        }

        static constexpr double signed_sqrt_ratio(double numerator, double denominator) noexcept
        {
            const double ratio = numerator / denominator;
            if (ratio < 0.0)
                return -math::sqrt(-ratio);
            return math::sqrt(ratio);
        }

        template <typename GeometryRuntime>
        constexpr double solve_pf_psin_beta_alpha1(const RadialVector&    relative_pressure,
                                                   const GeometryRuntime& geometry,
                                                   double                 p0,
                                                   double                 alpha2_per_alpha1,
                                                   double                 beta_target) const noexcept
        {
            const double volume = dot_radial_moment(geometry, geometry::radial_V_r);
            const double quadratic =
                alpha2_per_alpha1 *
                weighted_dot(relative_pressure, geometry, geometry::radial_V_r);
            const double linear = p0 * volume;
            if (math::abs(quadratic) <= 1.0e-14)
                return beta_target / linear;
            const double discriminant =
                linear * linear + 4.0 * quadratic * beta_target;
            const double root = math::sqrt(discriminant);
            const double root_plus = (-linear + root) / (2.0 * quadratic);
            const double root_minus = (-linear - root) / (2.0 * quadratic);
            const double preferred = signed_sqrt_ratio(beta_target, quadratic);
            if (preferred >= 0.0)
                return root_plus >= 0.0 ? root_plus : root_minus;
            return root_minus <= 0.0 ? root_minus : root_plus;
        }

        constexpr void regularize_ffn_psin(size_t n_axis_fix) noexcept
        {
            if (n_axis_fix == 0 || n_axis_fix + 1 >= radial_nodes)
                return;

            const size_t anchor0  = n_axis_fix;
            const size_t anchor1  = n_axis_fix + 1;
            const double x0       = GridType::nodes[anchor0] * GridType::nodes[anchor0];
            const double x1       = GridType::nodes[anchor1] * GridType::nodes[anchor1];
            const double value0   = FFn_psin[anchor0];
            const double value1   = FFn_psin[anchor1];
            const double inv_dx   = 1.0 / (x1 - x0);
            const double gradient = (value1 - value0) * inv_dx;

            for (size_t i = 0; i < n_axis_fix; ++i)
            {
                const double x = GridType::nodes[i] * GridType::nodes[i];
                FFn_psin[i]    = value0 + gradient * (x - x0);
            }
        }
    };
} // namespace source::detail

namespace source
{
    using detail::NativeSourceRuntime;
    using detail::SampledSourceShape;
    using detail::axis_fix_count;
    using detail::root_field_count;
    using detail::root_psin;
    using detail::root_psin_r;
    using detail::root_psin_rr;
} // namespace source
