#pragma once

#include "math.h"
#include "profiles.h"
#include "tensor.h"
#include <cstddef>
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
} // namespace source
