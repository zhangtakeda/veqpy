#pragma once

// Residual surface and root assembly helpers for generated Cxx Kernel artifacts.

#include "geometry.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"
#include <array>
#include <cstddef>
#include <span>

namespace residual::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    inline constexpr size_t block_h     = 0;
    inline constexpr size_t block_v     = 1;
    inline constexpr size_t block_kappa = 2;
    inline constexpr size_t block_c0    = 3;
    inline constexpr size_t block_c     = 4;
    inline constexpr size_t block_s     = 5;
    inline constexpr size_t block_psin  = 6;
    inline constexpr size_t block_F     = 7;

    template <typename Shape, typename GridType>
    struct ResidualRuntime
    {
        static_assert(Shape::L_max == GridType::basis_rows, "residual/profile basis rows must match");
        static_assert(Shape::K_max == GridType::rho_power_rows, "residual/profile rho rows must match");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "residual/profile harmonics must match");

        using shape = Shape;
        using grid  = GridType;

        static constexpr size_t radial_nodes = GridType::radial_nodes;
        static constexpr size_t theta_rows   = GridType::theta_rows;

        using MomentRows   = Matrix<double, Shape::active_count, radial_nodes>;
        using RadialVector = Vector<double, radial_nodes>;
        using PackedVector = Vector<double, Shape::x_size>;

        MomentRows moments{};

        constexpr void clear() noexcept { moments.clear(); }

        template <typename SourceRuntime, typename GeometryRuntime>
        constexpr void update_compact(const SourceRuntime&   source_runtime,
                                      const GeometryRuntime& geometry_runtime) noexcept
        {
            static_assert(SourceRuntime::radial_nodes == radial_nodes, "residual/source radial grids must match");
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "residual/geometry radial grids must match");
            static_assert(GeometryRuntime::theta_rows == theta_rows, "residual/geometry theta grids must match");

            update_fused(source_runtime, geometry_runtime);
        }

        constexpr void pack_into(std::span<double, Shape::x_size> out, double a, double R0, double B0) noexcept
        {
            pack_profile<0, 0>(out, a, R0, B0, nullptr);
        }

        constexpr void pack_scaled_into(std::span<double, Shape::x_size> out,
                                        double                           a,
                                        double                           R0,
                                        double                           B0,
                                        const double*                    output_scale) noexcept
        {
            pack_profile<0, 0>(out, a, R0, B0, output_scale);
        }

    private:
        template <typename SourceRuntime, typename GeometryRuntime>
        constexpr void update_fused(const SourceRuntime&   source_runtime,
                                    const GeometryRuntime& geometry_runtime) noexcept
        {
            const double alpha1 = source_runtime.alpha1;
            const double alpha2 = source_runtime.alpha2;

            constexpr size_t geometry_row_stride    = theta_rows;
            constexpr size_t geometry_radial_stride = geometry::surface_field_count * theta_rows;

            const double* const geometry_surface = geometry_runtime.surface_fields.aligned_data();

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                std::array<double, Shape::active_count> theta_moments{};
                const double psin_r_i             = source_runtime.profile_root_fields(source::root_psin_r, i);
                const double psin_rr_i            = source_runtime.profile_root_fields(source::root_psin_rr, i);
                const double FFn_psin_i           = source_runtime.FFn_psin[i];
                const double Pn_psin_i            = source_runtime.Pn_psin[i];
                const size_t geometry_radial_base = i * geometry_radial_stride;

                for (size_t j = 0; j < theta_rows; ++j)
                {
                    const size_t geometry_base = geometry_radial_base + j;
                    const double sin_tb_ij =
                        geometry_surface[geometry_base + geometry::surface_sin_tb * geometry_row_stride];
                    const double R2_ij = geometry_surface[geometry_base + geometry::surface_R2 * geometry_row_stride];
                    const double R_t_ij = geometry_surface[geometry_base + geometry::surface_R_t * geometry_row_stride];
                    const double Z_t_ij = geometry_surface[geometry_base + geometry::surface_Z_t * geometry_row_stride];
                    const double inv_J =
                        geometry_surface[geometry_base + geometry::surface_inv_J * geometry_row_stride];
                    const double JdivR_ij =
                        geometry_surface[geometry_base + geometry::surface_JdivR * geometry_row_stride];
                    const double G2_r_ij =
                        geometry_surface[geometry_base + geometry::surface_G2_r * geometry_row_stride];
                    const double gttdivJR_ij =
                        geometry_surface[geometry_base + geometry::surface_gttdivJR * geometry_row_stride];

                    const double psin_R = -Z_t_ij * inv_J * psin_r_i;
                    const double psin_Z = R_t_ij * inv_J * psin_r_i;

                    const double G1n            = JdivR_ij * (FFn_psin_i + R2_ij * Pn_psin_i);
                    const double G2n            = gttdivJR_ij * psin_rr_i + G2_r_ij * psin_r_i;
                    const double G_ij           = alpha1 * G1n + alpha2 * G2n;
                    const double Gpsin_R        = G_ij * psin_R;
                    const double Gpsin_Z        = G_ij * psin_Z;
                    const double Gpsin_R_sin_tb = Gpsin_R * sin_tb_ij;
                    accumulate_theta_moments<0>(theta_moments, j, G_ij, Gpsin_R, Gpsin_Z, Gpsin_R_sin_tb);
                }

                for (size_t active_index = 0; active_index < Shape::active_count; ++active_index)
                    moments(active_index, i) = theta_moments[active_index];
            }
        }

        template <size_t ProfileId, size_t ActiveIndex>
        constexpr void pack_profile(std::span<double, Shape::x_size> out,
                                    double                           a,
                                    double                           R0,
                                    double                           B0,
                                    const double*                    output_scale) noexcept
        {
            if constexpr (ProfileId < Shape::profile_count)
            {
                constexpr profiles::ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.optimized())
                {
                    pack_one<ProfileId, ActiveIndex, slot.coefficient_count>(out, a, R0, B0, output_scale);
                    pack_profile<ProfileId + 1, ActiveIndex + 1>(out, a, R0, B0, output_scale);
                }
                else
                    pack_profile<ProfileId + 1, ActiveIndex>(out, a, R0, B0, output_scale);
            }
        }

        template <size_t ProfileId, size_t ActiveIndex, size_t Count>
        constexpr void pack_one(std::span<double, Shape::x_size> out,
                                double                           a,
                                double                           R0,
                                double                           B0,
                                const double*                    output_scale) noexcept
        {
            constexpr size_t code         = block_code<ProfileId>();
            constexpr size_t radial_power = block_radial_power<ProfileId>();

            if constexpr (code == block_h)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out, GridType::y, unit_weights(), unit_weights(), a * base_scale(), output_scale);
            }
            else if constexpr (code == block_v)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out, GridType::y, unit_weights(), unit_weights(), a * base_scale(), output_scale);
            }
            else if constexpr (code == block_kappa)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out, rho_power<1>(), GridType::y, unit_weights(), -a * base_scale(), output_scale);
            }
            else if constexpr (code == block_c0)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out, rho_power<1>(), GridType::y, unit_weights(), -a * base_scale(), output_scale);
            }
            else if constexpr (code == block_c)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out,
                    rho_power<radial_power + 1>(),
                    GridType::y,
                    unit_weights(),
                    -a * base_scale(),
                    output_scale);
            }
            else if constexpr (code == block_s)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out,
                    rho_power<radial_power + 1>(),
                    GridType::y,
                    unit_weights(),
                    -a * base_scale(),
                    output_scale);
            }
            else if constexpr (code == block_psin)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out, rho_power<2>(), GridType::y, unit_weights(), base_scale(), output_scale);
            }
            else if constexpr (code == block_F)
            {
                project_scaled<Count, ProfileId, ActiveIndex>(
                    out,
                    GridType::y,
                    GridType::y,
                    unit_weights(),
                    base_scale() * (R0 * B0) * (R0 * B0),
                    output_scale);
            }
        }

        template <size_t ActiveIndex>
        static constexpr void accumulate_theta_moments(std::array<double, Shape::active_count>& theta_moments,
                                                       size_t                                   theta_node,
                                                       double                                   G,
                                                       double                                   Gpsin_R,
                                                       double                                   Gpsin_Z,
                                                       double                                   Gpsin_R_sin_tb) noexcept
        {
            if constexpr (ActiveIndex < Shape::active_count)
            {
                constexpr size_t profile_id = Shape::active_profile_ids[ActiveIndex];
                constexpr size_t code       = block_code<profile_id>();
                constexpr size_t order      = block_order<profile_id>();

                if constexpr (code == block_h)
                    theta_moments[ActiveIndex] += Gpsin_R;
                else if constexpr (code == block_v)
                    theta_moments[ActiveIndex] += Gpsin_Z;
                else if constexpr (code == block_kappa)
                    theta_moments[ActiveIndex] += Gpsin_Z * GridType::sin_mtheta(1, theta_node);
                else if constexpr (code == block_c0)
                    theta_moments[ActiveIndex] += Gpsin_R_sin_tb;
                else if constexpr (code == block_c)
                    theta_moments[ActiveIndex] += Gpsin_R_sin_tb * GridType::cos_mtheta(order, theta_node);
                else if constexpr (code == block_s)
                    theta_moments[ActiveIndex] += Gpsin_R_sin_tb * GridType::sin_mtheta(order, theta_node);
                else
                    theta_moments[ActiveIndex] += G;

                accumulate_theta_moments<ActiveIndex + 1>(
                    theta_moments, theta_node, G, Gpsin_R, Gpsin_Z, Gpsin_R_sin_tb);
            }
        }

        static constexpr double base_scale() noexcept
        {
            return 2.0 * geometry::detail::pi / static_cast<double>(theta_rows);
        }

        template <size_t ProfileId>
        static consteval size_t block_code()
        {
            if constexpr (ProfileId == Shape::h_profile_id)
                return block_h;
            else if constexpr (ProfileId == Shape::v_profile_id)
                return block_v;
            else if constexpr (ProfileId == Shape::kappa_profile_id)
                return block_kappa;
            else if constexpr (ProfileId == Shape::c0_profile_id)
                return block_c0;
            else if constexpr (ProfileId > Shape::c0_profile_id && ProfileId <= Shape::c_profile_id(Shape::M_max))
                return block_c;
            else if constexpr (ProfileId >= Shape::s_profile_id(1) && ProfileId <= Shape::s_profile_id(Shape::M_max))
                return block_s;
            else if constexpr (ProfileId == Shape::psin_profile_id)
                return block_psin;
            else
                return block_F;
        }

        template <size_t ProfileId>
        static consteval size_t block_order()
        {
            if constexpr (ProfileId > Shape::c0_profile_id && ProfileId <= Shape::c_profile_id(Shape::M_max))
                return ProfileId - Shape::c0_profile_id;
            else if constexpr (ProfileId >= Shape::s_profile_id(1) && ProfileId <= Shape::s_profile_id(Shape::M_max))
                return ProfileId - Shape::c0_profile_id - Shape::M_max;
            else
                return 0;
        }

        template <size_t ProfileId>
        static consteval size_t block_radial_power()
        {
            constexpr size_t order = block_order<ProfileId>();
            if constexpr (order == 0)
                return 0;
            else
                return order < GridType::rho_power_rows ? order : GridType::rho_power_rows;
        }

        template <size_t Count,
                  size_t ProfileId,
                  size_t ActiveIndex,
                  typename WeightA,
                  typename WeightB,
                  typename WeightC>
        constexpr void project_scaled(std::span<double, Shape::x_size> out,
                                      const WeightA&                    weight_a,
                                      const WeightB&                    weight_b,
                                      const WeightC&                    weight_c,
                                      double                            scalar,
                                      const double*                     output_scale) const noexcept
        {
            for (size_t degree = 0; degree < Count; ++degree)
            {
                double total = 0.0;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    total += projection_basis(degree, i) * moments(ActiveIndex, i) * weight_a[i] * weight_b[i] *
                             weight_c[i] * GridType::weights[i];
                }
                const size_t output_index = static_cast<size_t>(Shape::coeff_index[ProfileId][degree]);
                if (output_scale == nullptr)
                    out[output_index] = total * scalar;
#if defined(VEQPY_CXX_FP_MODE_RELAXED)
                else
                {
                    // Preserve the raw residual rounding point under -ffast-math. Reassociation here can alter the
                    // finite-difference LM trajectory even when the final scaled values remain within tolerance.
                    const volatile double raw_value = total * scalar;
                    out[output_index] = raw_value * output_scale[output_index];
                }
#else
                else
                {
                    const double raw_value = total * scalar;
                    out[output_index] = raw_value / output_scale[output_index];
                }
#endif
            }
        }

        constexpr double projection_basis(size_t degree, size_t node) const noexcept
        {
            return degree == 0 ? 1.0 : GridType::T(degree - 1, node);
        }

        static constexpr RadialVector unit_weights() noexcept
        {
            RadialVector out{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
                out[i] = 1.0;
            return out;
        }

        template <size_t Power>
        static constexpr RadialVector rho_power() noexcept
        {
            RadialVector out{uninitialized};
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                double value = 1.0;
                for (size_t k = 0; k < Power; ++k)
                    value *= GridType::nodes[i];
                out[i] = value;
            }
            return out;
        }
    };
} // namespace residual::detail

namespace residual
{
    using detail::ResidualRuntime;
} // namespace residual
