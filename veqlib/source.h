#pragma once

#include "geometry.h"
#include "math.h"
#include "profiles.h"
#include "tensor.h"
#include <cstddef>
#include <limits>
#include <span>

namespace source::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    inline constexpr size_t default_barycentric_stencil = 8;

    inline constexpr size_t root_psin       = 0;
    inline constexpr size_t root_psin_r     = 1;
    inline constexpr size_t root_psin_rr    = 2;
    inline constexpr size_t root_field_count = 3;

    inline constexpr size_t profile_value   = 0;
    inline constexpr size_t profile_radial  = 1;
    inline constexpr size_t profile_radial2 = 2;

    constexpr double unset_constraint() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

    constexpr bool constraint_is_set(double value) noexcept { return value == value; }

    constexpr size_t clipped_stencil_size(size_t sample_count) noexcept
    {
        return sample_count < default_barycentric_stencil ? sample_count : default_barycentric_stencil;
    }

    template <size_t SampleCount, size_t StencilSize = clipped_stencil_size(SampleCount)>
    struct UniformSourceShape
    {
        static_assert(SampleCount >= 1, "uniform source requires at least one sample");
        static_assert(StencilSize >= 1, "uniform source barycentric stencil must be positive");
        static_assert(StencilSize <= SampleCount, "uniform source barycentric stencil exceeds sample count");

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
    constexpr size_t axis_fix_count(double fix_rho) noexcept
    {
        size_t count = 0;
        while (count < GridType::radial_nodes && GridType::nodes[count] < fix_rho)
            ++count;
        return count;
    }

    template <typename MatrixType, typename InVector, typename OutVector>
    constexpr void matvec_into(OutVector& out, const MatrixType& matrix, const InVector& values) noexcept
    {
        for (size_t row = 0; row < OutVector::shape[0]; ++row)
        {
            double total = 0.0;
            for (size_t col = 0; col < InVector::shape[0]; ++col)
                total += matrix(row, col) * values[col];
            out[row] = total;
        }
    }

    template <typename GridType, typename SourceShape>
    struct ProfileOwnedPsinSourceRuntime
    {
        static constexpr size_t radial_nodes = GridType::radial_nodes;
        static constexpr size_t sample_count = SourceShape::sample_count;
        static constexpr size_t stencil_size = SourceShape::stencil_size;

        using RadialVector = Vector<double, radial_nodes>;
        using SourceVector = Vector<double, sample_count>;
        using RootFields   = Matrix<double, root_field_count, radial_nodes>;

        SourceVector heat_input{};
        SourceVector current_input{};
        RadialVector source_psin_query{};
        RadialVector source_parameter_query{};
        RadialVector materialized_heat_input{};
        RadialVector materialized_current_input{};
        RootFields   source_target_root_fields{};
        RadialVector FFn_psin{};
        RadialVector Pn_psin{};
        double       alpha1 = unset_constraint();
        double       alpha2 = unset_constraint();

        constexpr void set_uniform_sources(std::span<const double, sample_count> heat,
                                           std::span<const double, sample_count> current) noexcept
        {
            for (size_t i = 0; i < sample_count; ++i)
            {
                heat_input[i]    = heat[i];
                current_input[i] = current[i];
            }
        }

        template <typename ProfilesRuntime>
        constexpr bool materialize_profile_owned_psin(const ProfilesRuntime& runtime_profiles,
                                                      size_t                 n_axis_fix) noexcept
        {
            using Shape       = typename ProfilesRuntime::shape;
            using ProfileGrid = typename ProfilesRuntime::grid;

            static_assert(ProfileGrid::radial_nodes == radial_nodes, "source/profile radial grids must match");
            static_assert(Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                          "PF psin-uniform materialization requires an active psin profile");

            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(root_psin_r, i) =
                    runtime_profiles.profile_field(Shape::psin_profile_id, i, profile_radial);

            regularize_psin_r(n_axis_fix);
            RadialVector psin_rr{uninitialized};
            matvec_into(psin_rr, GridType::differentiator, const_root_row<root_psin_r>());
            store_root_row<root_psin_rr>(psin_rr);
            if (!update_psin_coordinate())
                return false;

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double psin_value = source_target_root_fields(root_psin, i);
                source_psin_query[i]    = psin_value;
                source_parameter_query[i] = psin_value;
            }

            local_barycentric_interpolate_pair();
            return math::is_finite(source_target_root_fields) && math::is_finite(materialized_heat_input) &&
                   math::is_finite(materialized_current_input);
        }

        template <typename GeometryRuntime>
        constexpr bool update_pf_from_psin_uniform(const GeometryRuntime& geometry,
                                                   double                 B0,
                                                   double                 Ip,
                                                   double                 beta,
                                                   size_t                 n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");

            const bool has_Ip   = constraint_is_set(Ip);
            const bool has_beta = constraint_is_set(beta);
            if (has_Ip && has_beta)
                return false;

            RadialVector integrand{uninitialized};
            fill_pf_psin_integrand(integrand, geometry);

            RadialVector psin_r{uninitialized};
            matvec_into(psin_r, GridType::accumulator, integrand);
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                psin_r[i] *= -1.0;
                psin_r[i] /= geometry.radial_field(geometry::radial_Kn, i);
            }

            const double psi_scale_sign = weighted_profile_sign(psin_r);
            if (psi_scale_sign < 0.0)
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] *= -1.0;

            store_root_row<root_psin_r>(psin_r);
            regularize_psin_r(n_axis_fix);
            psin_r = const_root_row<root_psin_r>();

            const double integral_prof = dot(psin_r, GridType::weights);
            if (math::abs(integral_prof) < 1.0e-14)
                return false;

            for (size_t i = 0; i < radial_nodes; ++i)
                psin_r[i] /= integral_prof;
            store_root_row<root_psin_r>(psin_r);

            RadialVector psin_rr{uninitialized};
            matvec_into(psin_rr, GridType::differentiator, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
            if (!update_psin_coordinate())
                return false;

            if (!has_Ip && !has_beta)
            {
                alpha2 = psi_scale_sign * integral_prof;
                RadialVector pressure_profile{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                    pressure_profile[i] = materialized_heat_input[i] * psin_r[i];
                alpha1 = -dot(pressure_profile, GridType::weights);
                if (math::abs(alpha1) < 1.0e-14)
                    return false;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    Pn_psin[i]  = materialized_heat_input[i] / alpha1;
                    FFn_psin[i] = materialized_current_input[i] / alpha1;
                }
                regularize_ffn_psin(n_axis_fix);
                return math::is_finite(FFn_psin) && math::is_finite(Pn_psin) && math::is_finite(alpha1) &&
                       math::is_finite(alpha2);
            }

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                Pn_psin[i]  = materialized_heat_input[i];
                FFn_psin[i] = materialized_current_input[i];
            }
            regularize_ffn_psin(n_axis_fix);

            if (has_Ip)
            {
                const double G1n_integral = g1n_psin_integral_from_radial_moments(geometry);
                if (math::abs(G1n_integral) < 1.0e-14)
                    return false;
                alpha1 = -Ip / G1n_integral;
            }
            else
            {
                RadialVector scratch_Pn_r{uninitialized};
                for (size_t i = 0; i < radial_nodes; ++i)
                    scratch_Pn_r[i] = Pn_psin[i] * psin_r[i];

                RadialVector Pn_out{uninitialized};
                compute_Pn_out(Pn_out, scratch_Pn_r);

                const double numerator = 0.5 * beta * B0 * B0 * dot_geometry_radial(geometry, geometry::radial_V_r);
                const double denominator = weighted_dot(Pn_out, geometry, geometry::radial_V_r);
                if (math::abs(denominator) < 1.0e-14)
                    return false;
                alpha1 = signed_sqrt_ratio(numerator, integral_prof * denominator);
            }
            alpha2 = integral_prof * alpha1;
            return math::is_finite(FFn_psin) && math::is_finite(Pn_psin) && math::is_finite(alpha1) &&
                   math::is_finite(alpha2);
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
                const size_t anchor0 = n_axis_fix;
                const size_t anchor1 = n_axis_fix + 1;
                const double rho0    = GridType::nodes[anchor0];
                const double rho1    = GridType::nodes[anchor1];
                const double x0      = rho0 * rho0;
                const double x1      = rho1 * rho1;
                const double slope0  = source_target_root_fields(root_psin_r, anchor0) / rho0;
                const double slope1  = source_target_root_fields(root_psin_r, anchor1) / rho1;
                const double gradient = (slope1 - slope0) / (x1 - x0);

                for (size_t i = 0; i < n_axis_fix; ++i)
                {
                    const double rho_i = GridType::nodes[i];
                    const double x_i   = rho_i * rho_i;
                    source_target_root_fields(root_psin_r, i) = rho_i * (slope0 + gradient * (x_i - x0));
                }
            }

            for (size_t i = 0; i < radial_nodes; ++i)
                if (source_target_root_fields(root_psin_r, i) < 1.0e-10)
                    source_target_root_fields(root_psin_r, i) = 1.0e-10;
        }

        constexpr bool update_psin_coordinate() noexcept
        {
            const auto psin_r = const_root_row<root_psin_r>();
            RadialVector integrated{uninitialized};
            matvec_into(integrated, GridType::accumulator, psin_r);

            const double offset = integrated[0];
            const double scale  = integrated[radial_nodes - 1] - offset;
            if (math::abs(scale) < 1.0e-12)
                return false;

            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(root_psin, i) = (integrated[i] - offset) / scale;
            source_target_root_fields(root_psin, 0) = 0.0;
            source_target_root_fields(root_psin, radial_nodes - 1) = 1.0;
            return true;
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
                const double pos = q * static_cast<double>(sample_count - 1);
                size_t       center = static_cast<size_t>(pos);
                if (pos > static_cast<double>(center))
                    ++center;

                constexpr size_t half = stencil_size / 2;
                if (center < half)
                    return 0;

                const size_t start = center - half;
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
                    materialized_heat_input[i]    = heat_input[0];
                    materialized_current_input[i] = current_input[0];
                }
            }
            else
            {
                constexpr double denom_scale = static_cast<double>(sample_count - 1);
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    const double q = clip_unit(source_parameter_query[i]);
                    const size_t start = local_uniform_stencil_start(q);
                    size_t       hit   = sample_count;
                    for (size_t local_j = 0; local_j < stencil_size; ++local_j)
                    {
                        const size_t j  = start + local_j;
                        const double xj = static_cast<double>(j) / denom_scale;
                        if (math::abs(q - xj) <= 1.0e-14)
                        {
                            hit = j;
                            break;
                        }
                    }

                    if (hit < sample_count)
                    {
                        materialized_heat_input[i]    = heat_input[hit];
                        materialized_current_input[i] = current_input[hit];
                        continue;
                    }

                    double denominator = 0.0;
                    double numerator_heat = 0.0;
                    double numerator_current = 0.0;
                    for (size_t local_j = 0; local_j < stencil_size; ++local_j)
                    {
                        const size_t j = start + local_j;
                        const double term =
                            SourceShape::barycentric_weights[local_j] / (q - static_cast<double>(j) / denom_scale);
                        denominator += term;
                        numerator_heat += term * heat_input[j];
                        numerator_current += term * current_input[j];
                    }
                    materialized_heat_input[i]    = numerator_heat / denominator;
                    materialized_current_input[i] = numerator_current / denominator;
                }
            }
        }

        template <typename GeometryRuntime>
        constexpr void fill_pf_psin_integrand(RadialVector& out, const GeometryRuntime& geometry) const noexcept
        {
            constexpr double pressure_factor = 1.0 / (4.0 * geometry::detail::pi * geometry::detail::pi);
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                out[i] = materialized_current_input[i] * geometry.radial_field(geometry::radial_Ln_r, i) +
                         geometry.radial_field(geometry::radial_V_r, i) * materialized_heat_input[i] *
                             pressure_factor;
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

        constexpr double weighted_profile_sign(const RadialVector& values) const noexcept
        {
            return dot(values, GridType::weights) < 0.0 ? -1.0 : 1.0;
        }

        template <typename GeometryRuntime>
        constexpr double dot_geometry_radial(const GeometryRuntime& geometry, size_t row) const noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += GridType::weights[i] * geometry.radial_field(row, i);
            return total;
        }

        template <typename GeometryRuntime>
        constexpr double weighted_dot(const RadialVector& lhs, const GeometryRuntime& geometry, size_t row) const noexcept
        {
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
                total += GridType::weights[i] * lhs[i] * geometry.radial_field(row, i);
            return total;
        }

        static constexpr double signed_sqrt_ratio(double numerator, double denominator) noexcept
        {
            const double ratio = numerator / denominator;
            return ratio < 0.0 ? -math::sqrt(-ratio) : math::sqrt(ratio);
        }

        constexpr void compute_Pn_out(RadialVector& out, const RadialVector& Pn_r) const noexcept
        {
            matvec_into(out, GridType::accumulator, Pn_r);
            const double offset = dot(Pn_r, GridType::weights);
            for (size_t i = 0; i < radial_nodes; ++i)
                out[i] -= offset;
        }

        template <typename GeometryRuntime>
        constexpr double g1n_psin_integral_from_radial_moments(const GeometryRuntime& geometry) const noexcept
        {
            constexpr double two_pi = 2.0 * geometry::detail::pi;
            constexpr double inv_two_pi = 1.0 / two_pi;
            double total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                total += GridType::weights[i] *
                         (two_pi * geometry.radial_field(geometry::radial_Ln_r, i) * FFn_psin[i] +
                          inv_two_pi * geometry.radial_field(geometry::radial_V_r, i) * Pn_psin[i]);
            }
            return total;
        }

        constexpr void regularize_ffn_psin(size_t n_axis_fix) noexcept
        {
            if (n_axis_fix == 0 || n_axis_fix + 1 >= radial_nodes)
                return;

            const size_t anchor0 = n_axis_fix;
            const size_t anchor1 = n_axis_fix + 1;
            const double x0      = GridType::nodes[anchor0] * GridType::nodes[anchor0];
            const double x1      = GridType::nodes[anchor1] * GridType::nodes[anchor1];
            const double value0  = FFn_psin[anchor0];
            const double value1  = FFn_psin[anchor1];
            const double gradient = (value1 - value0) / (x1 - x0);

            for (size_t i = 0; i < n_axis_fix; ++i)
            {
                const double x = GridType::nodes[i] * GridType::nodes[i];
                FFn_psin[i] = value0 + gradient * (x - x0);
            }
        }
    };
} // namespace source::detail

namespace source
{
    using detail::ProfileOwnedPsinSourceRuntime;
    using detail::UniformSourceShape;
    using detail::axis_fix_count;
    using detail::root_field_count;
    using detail::root_psin;
    using detail::root_psin_r;
    using detail::root_psin_rr;
    using detail::unset_constraint;
} // namespace source
