#pragma once

#include "geometry.h"
#include "math.h"
#include "tensor.h"
#include "tensor_kernels.h"
#include "tensor_layout.h"
#include <cstddef>
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
    struct PfPsinUniformIpSourceRuntime
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
        RootFields   profile_root_fields{};
        RootFields   source_target_root_fields{};
        RadialVector FFn_psin{};
        RadialVector Pn_psin{};
        double       alpha1 = 0.0;
        double       alpha2 = 0.0;

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
        constexpr void materialize_profile_owned_psin(const ProfilesRuntime& runtime_profiles,
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
            }

            local_barycentric_interpolate_pair();
        }

        template <typename GeometryRuntime>
        constexpr void update_pf_psin_uniform_ip(const GeometryRuntime& geometry, double Ip, size_t n_axis_fix) noexcept
        {
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "source/geometry radial grids must match");

            RadialVector integrand{uninitialized};
            fill_pf_psin_integrand(integrand, geometry);

            RadialVector psin_r{uninitialized};
            radial_grid_accumulator_matvec_into<GridType>(psin_r, integrand);
            double psin_r_weighted_total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                psin_r[i] *= -1.0;
                psin_r[i] /= geometry.radial_field(geometry::radial_Kn, i);
                psin_r_weighted_total += psin_r[i] * GridType::weights[i];
            }

            if (psin_r_weighted_total < 0.0)
                for (size_t i = 0; i < radial_nodes; ++i)
                    psin_r[i] *= -1.0;

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

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                Pn_psin[i]  = materialized_heat_input[i];
                FFn_psin[i] = materialized_current_input[i];
            }
            regularize_ffn_psin(n_axis_fix);

            const double G1n_integral = g1n_psin_integral_from_radial_moments(geometry);
            alpha1                    = -Ip / G1n_integral;
            alpha2                    = integral_prof * alpha1;
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

        template <typename ProfilesRuntime>
        constexpr void benchmark_copy_profile_psin_r(const ProfilesRuntime& runtime_profiles) noexcept
        {
            using Shape       = typename ProfilesRuntime::shape;
            using ProfileGrid = typename ProfilesRuntime::grid;

            static_assert(ProfileGrid::radial_nodes == radial_nodes, "source/profile radial grids must match");
            static_assert(Shape::slot_for_profile_id(Shape::psin_profile_id).optimized(),
                          "PF psin-uniform materialization requires an active psin profile");

            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(root_psin_r, i) =
                    runtime_profiles.profile_field(Shape::psin_profile_id, i, profile_radial);
        }

        constexpr void benchmark_regularize_psin_r(size_t n_axis_fix) noexcept { regularize_psin_r(n_axis_fix); }

        constexpr void benchmark_D_psin_into_rr() noexcept
        {
            RadialVector psin_rr{uninitialized};
            matvec_into(psin_rr, GridType::differentiator, const_root_row<root_psin_r>());
            store_root_row<root_psin_rr>(psin_rr);
        }

        constexpr void benchmark_A_psin_into(RadialVector& out) const noexcept
        {
            matvec_into(out, GridType::accumulator, const_root_row<root_psin_r>());
        }

        constexpr void benchmark_DA_psin_into(RadialVector& psin_rr, RadialVector& integrated) const noexcept
        {
            multi_matvec_into(
                psin_rr, integrated, GridType::differentiator, GridType::accumulator, const_root_row<root_psin_r>());
        }

        void benchmark_DA_psin_packed_into(RadialVector& psin_rr, RadialVector& integrated) const noexcept
        {
            radial_grid_multi_matvec_into<GridType>(psin_rr, integrated, const_root_row<root_psin_r>());
        }

        constexpr void benchmark_prepare_psin_queries() noexcept
        {
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double psin_value   = source_target_root_fields(root_psin, i);
                source_psin_query[i]      = psin_value;
                source_parameter_query[i] = psin_value;
            }
        }

        constexpr void benchmark_interpolate_pair() noexcept { local_barycentric_interpolate_pair(); }

        template <typename GeometryRuntime>
        constexpr void benchmark_fill_pf_psin_integrand(RadialVector&          out,
                                                        const GeometryRuntime& geometry) const noexcept
        {
            fill_pf_psin_integrand(out, geometry);
        }

        constexpr void benchmark_A_integrand_into(RadialVector& out, const RadialVector& integrand) const noexcept
        {
            radial_grid_accumulator_matvec_into<GridType>(out, integrand);
        }

        constexpr void benchmark_A_integrand_rowdot_into(RadialVector&       out,
                                                         const RadialVector& integrand) const noexcept
        {
            matvec_into(out, GridType::accumulator, integrand);
        }

        template <typename GeometryRuntime>
        constexpr double benchmark_normalize_psin_r_into(RadialVector&          out,
                                                         const RadialVector&    integrated,
                                                         const GeometryRuntime& geometry,
                                                         size_t                 n_axis_fix) noexcept
        {
            out                          = integrated;
            double psin_r_weighted_total = 0.0;
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                out[i] *= -1.0;
                out[i] /= geometry.radial_field(geometry::radial_Kn, i);
                psin_r_weighted_total += out[i] * GridType::weights[i];
            }

            if (psin_r_weighted_total < 0.0)
                for (size_t i = 0; i < radial_nodes; ++i)
                    out[i] *= -1.0;

            store_root_row<root_psin_r>(out);
            regularize_psin_r(n_axis_fix);
            out = const_root_row<root_psin_r>();

            const double integral_prof = dot(out, GridType::weights);
            for (size_t i = 0; i < radial_nodes; ++i)
                out[i] /= integral_prof;
            store_root_row<root_psin_r>(out);
            return integral_prof;
        }

        constexpr void benchmark_D_normalized_psin_into_rr(const RadialVector& psin_r) noexcept
        {
            RadialVector psin_rr{uninitialized};
            matvec_into(psin_rr, GridType::differentiator, psin_r);
            store_root_row<root_psin_rr>(psin_rr);
        }

        template <typename GeometryRuntime>
        constexpr void
        benchmark_update_alpha_from_integral(const GeometryRuntime& geometry, double Ip, double integral_prof) noexcept
        {
            const double G1n_integral = g1n_psin_integral_from_radial_moments(geometry);
            alpha1                    = -Ip / G1n_integral;
            alpha2                    = integral_prof * alpha1;
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
                const double slope0   = source_target_root_fields(root_psin_r, anchor0) / rho0;
                const double slope1   = source_target_root_fields(root_psin_r, anchor1) / rho1;
                const double gradient = (slope1 - slope0) / (x1 - x0);

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
            for (size_t i = 0; i < radial_nodes; ++i)
                source_target_root_fields(root_psin, i) = (integrated[i] - offset) / scale;
            source_target_root_fields(root_psin, 0)                = 0.0;
            source_target_root_fields(root_psin, radial_nodes - 1) = 1.0;
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
                    materialized_heat_input[i]    = heat_input[0];
                    materialized_current_input[i] = current_input[0];
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
                        materialized_heat_input[i]    = heat_input[nearest];
                        materialized_current_input[i] = current_input[nearest];
                        continue;
                    }

                    double denominator       = 0.0;
                    double numerator_heat    = 0.0;
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
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                out[i] = materialized_current_input[i] * geometry.radial_field(geometry::radial_Ln_r, i) +
                         geometry.radial_field(geometry::radial_V_r, i) * materialized_heat_input[i] * pressure_factor;
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
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                total +=
                    GridType::weights[i] * (two_pi * geometry.radial_field(geometry::radial_Ln_r, i) * FFn_psin[i] +
                                            inv_two_pi * geometry.radial_field(geometry::radial_V_r, i) * Pn_psin[i]);
            }
            return total;
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
            const double gradient = (value1 - value0) / (x1 - x0);

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
    using detail::PfPsinUniformIpSourceRuntime;
    using detail::UniformSourceShape;
    using detail::axis_fix_count;
    using detail::root_field_count;
    using detail::root_psin;
    using detail::root_psin_r;
    using detail::root_psin_rr;
} // namespace source
