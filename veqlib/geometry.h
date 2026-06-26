#pragma once

#include "math.h"
#include "tensor.h"
#include <array>
#include <cstddef>

namespace geometry::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Tensor;

    inline constexpr double pi      = 3.141592653589793238462643383279502884;

    inline constexpr size_t surface_sin_tb      = 0;
    inline constexpr size_t surface_R           = 1;
    inline constexpr size_t surface_R_t         = 2;
    inline constexpr size_t surface_Z_t         = 3;
    inline constexpr size_t surface_J           = 4;
    inline constexpr size_t surface_JdivR       = 5;
    inline constexpr size_t surface_grtdivJR_t  = 6;
    inline constexpr size_t surface_gttdivJR    = 7;
    inline constexpr size_t surface_gttdivJR_r  = 8;
    inline constexpr size_t surface_field_count = 9;

    inline constexpr size_t radial_S_r         = 0;
    inline constexpr size_t radial_V_r         = 1;
    inline constexpr size_t radial_Kn          = 2;
    inline constexpr size_t radial_Kn_r        = 3;
    inline constexpr size_t radial_Ln_r        = 4;
    inline constexpr size_t radial_field_count = 5;

    inline constexpr size_t profile_value   = 0;
    inline constexpr size_t profile_radial  = 1;
    inline constexpr size_t profile_radial2 = 2;

    template <typename GridType>
    struct GeometryRuntime
    {
        static constexpr size_t radial_nodes = GridType::radial_nodes;
        static constexpr size_t theta_rows   = GridType::theta_rows;

        // Store [rho][field][theta]: theta stays contiguous while field starts no longer
        // alias at the default 4096 B plane stride.
        using SurfaceSlab = Tensor<double, radial_nodes, surface_field_count, theta_rows>;
        using RadialSlab  = Matrix<double, radial_field_count, radial_nodes>;

        SurfaceSlab surface_fields{};
        RadialSlab  radial_fields{};

        constexpr void clear() noexcept
        {
            surface_fields.clear();
            radial_fields.clear();
        }

        constexpr double& surface_field(size_t row, size_t radial_node, size_t theta_node) noexcept
        {
            return surface_fields(radial_node, row, theta_node);
        }

        constexpr double surface_field(size_t row, size_t radial_node, size_t theta_node) const noexcept
        {
            return surface_fields(radial_node, row, theta_node);
        }

        constexpr double& radial_field(size_t row, size_t radial_node) noexcept
        {
            return radial_fields(row, radial_node);
        }

        constexpr double radial_field(size_t row, size_t radial_node) const noexcept
        {
            return radial_fields(row, radial_node);
        }

        template <typename ProfilesRuntime>
        constexpr void update(double a, double R0, double Z0, const ProfilesRuntime& runtime_profiles) noexcept
        {
            using Shape       = typename ProfilesRuntime::shape;
            using ProfileGrid = typename ProfilesRuntime::grid;

            static_assert(ProfileGrid::radial_nodes == radial_nodes, "geometry/profile radial grids must match");
            static_assert(ProfileGrid::theta_rows == theta_rows, "geometry/profile theta grids must match");
            static_assert(ProfileGrid::harmonic_rows == GridType::harmonic_rows,
                          "geometry/profile harmonics must match");

            (void)Z0;

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double rho_i  = GridType::nodes[i];
                const double h_i    = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_value);
                const double h_r_i  = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial);
                const double h_rr_i = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial2);
                const double v_r_i  = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial);
                const double v_rr_i = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial2);
                const double k_i    = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_value);
                const double k_r_i  = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial);
                const double k_rr_i = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial2);

                double sum_J          = 0.0;
                double sum_JR         = 0.0;
                double sum_gttdivJR   = 0.0;
                double sum_gttdivJR_r = 0.0;
                double sum_JdivR      = 0.0;

                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_r_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_t_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_rr_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_rt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> tb_tt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> sin_tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, theta_rows> cos_tb_values;

                for (size_t j = 0; j < theta_rows; ++j)
                {
                    double tb_ij    = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb);
                    double tb_r_ij  = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_r);
                    double tb_t_ij  = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_t);
                    double tb_rr_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_rr);
                    double tb_rt_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_rt);
                    double tb_tt_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_tt);

                    for (size_t active_index = 0; active_index < Shape::active_c_order_count; ++active_index)
                    {
                        const size_t order  = Shape::active_c_orders[active_index];
                        const double c_i    = runtime_profiles.c_family_fields(order, i, profile_value);
                        const double c_r_i  = runtime_profiles.c_family_fields(order, i, profile_radial);
                        const double c_rr_i = runtime_profiles.c_family_fields(order, i, profile_radial2);

                        if (order == 0)
                        {
                            tb_ij += c_i;
                            tb_r_ij += c_r_i;
                            tb_rr_ij += c_rr_i;
                            continue;
                        }

                        const double cos_kt    = GridType::cos_mtheta(order, j);
                        const double k_sin_kt  = GridType::m_sin_mtheta(order, j);
                        const double k2_cos_kt = GridType::m2_cos_mtheta(order, j);

                        tb_ij += c_i * cos_kt;
                        tb_r_ij += c_r_i * cos_kt;
                        tb_t_ij -= c_i * k_sin_kt;
                        tb_rr_ij += c_rr_i * cos_kt;
                        tb_rt_ij -= c_r_i * k_sin_kt;
                        tb_tt_ij -= c_i * k2_cos_kt;
                    }

                    for (size_t active_index = 0; active_index < Shape::active_s_order_count; ++active_index)
                    {
                        const size_t order     = Shape::active_s_orders[active_index];
                        const double s_i       = runtime_profiles.s_family_fields(order, i, profile_value);
                        const double s_r_i     = runtime_profiles.s_family_fields(order, i, profile_radial);
                        const double s_rr_i    = runtime_profiles.s_family_fields(order, i, profile_radial2);
                        const double sin_kt    = GridType::sin_mtheta(order, j);
                        const double k_cos_kt  = GridType::m_cos_mtheta(order, j);
                        const double k2_sin_kt = GridType::m2_sin_mtheta(order, j);

                        tb_ij += s_i * sin_kt;
                        tb_r_ij += s_r_i * sin_kt;
                        tb_t_ij += s_i * k_cos_kt;
                        tb_rr_ij += s_rr_i * sin_kt;
                        tb_rt_ij += s_r_i * k_cos_kt;
                        tb_tt_ij -= s_i * k2_sin_kt;
                    }

                    tb_values[j]    = tb_ij;
                    tb_r_values[j]  = tb_r_ij;
                    tb_t_values[j]  = tb_t_ij;
                    tb_rr_values[j] = tb_rr_ij;
                    tb_rt_values[j] = tb_rt_ij;
                    tb_tt_values[j] = tb_tt_ij;
                }

#pragma clang loop vectorize(enable)
                for (size_t j = 0; j < theta_rows; ++j)
                {
                    const double tb_ij = tb_values[j];
                    math::relaxed_sincos(tb_ij, sin_tb_values[j], cos_tb_values[j]);
                }

                for (size_t j = 0; j < theta_rows; ++j)
                {
                    const double sin_t = GridType::sin_mtheta(1, j);
                    const double cos_t = GridType::cos_mtheta(1, j);

                    const double tb_r_ij   = tb_r_values[j];
                    const double tb_t_ij   = tb_t_values[j];
                    const double tb_rr_ij  = tb_rr_values[j];
                    const double tb_rt_ij  = tb_rt_values[j];
                    const double tb_tt_ij  = tb_tt_values[j];
                    const double cos_tb_ij = cos_tb_values[j];
                    const double sin_tb_ij = sin_tb_values[j];

                    double R_ij = R0 + a * (h_i + rho_i * cos_tb_ij);
                    if (R_ij < 1.0e-6)
                        R_ij = 1.0e-6;

                    const double R_r_ij  = a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij);
                    const double R_t_ij  = -a * rho_i * sin_tb_ij * tb_t_ij;
                    const double R_rr_ij = a * (h_rr_i - 2.0 * sin_tb_ij * tb_r_ij -
                                                rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij));
                    const double R_rt_ij =
                        -a * (sin_tb_ij * tb_t_ij + rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij));
                    const double R_tt_ij = -a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij);

                    const double Z_r_ij  = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t);
                    const double Z_t_ij  = -a * rho_i * k_i * cos_t;
                    const double Z_rr_ij = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t);
                    const double Z_rt_ij = -a * (k_i + rho_i * k_r_i) * cos_t;
                    const double Z_tt_ij = a * rho_i * k_i * sin_t;

                    double J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij;
                    if (J_ij < 1.0e-6)
                        J_ij = 1.0e-6;

                    const double J_r_ij  = -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij);
                    const double J_t_ij  = -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij);
                    const double JR_ij   = J_ij * R_ij;
                    const double JR_r_ij = J_r_ij * R_ij + J_ij * R_r_ij;
                    const double JR_t_ij = J_t_ij * R_ij + J_ij * R_t_ij;
                    const double JdivR_ij = J_ij / R_ij;
                    const double grt_ij   = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij;
                    const double grt_t_ij = R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij;
                    const double gtt_ij   = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij;
                    const double gtt_r_ij = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij);
                    const double inv_JR   = 1.0 / JR_ij;
                    const double grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR;
                    const double gttdivJR_ij   = gtt_ij * inv_JR;
                    const double gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR;

                    surface_field(surface_sin_tb, i, j)     = sin_tb_ij;
                    surface_field(surface_R, i, j)          = R_ij;
                    surface_field(surface_R_t, i, j)        = R_t_ij;
                    surface_field(surface_Z_t, i, j)        = Z_t_ij;
                    surface_field(surface_J, i, j)          = J_ij;
                    surface_field(surface_JdivR, i, j)      = JdivR_ij;
                    surface_field(surface_grtdivJR_t, i, j) = grtdivJR_t_ij;
                    surface_field(surface_gttdivJR, i, j)   = gttdivJR_ij;
                    surface_field(surface_gttdivJR_r, i, j) = gttdivJR_r_ij;

                    sum_J += J_ij;
                    sum_JR += JR_ij;
                    sum_gttdivJR += gttdivJR_ij;
                    sum_gttdivJR_r += gttdivJR_r_ij;
                    sum_JdivR += JdivR_ij;
                }

                constexpr double theta_scale = 2.0 * pi / static_cast<double>(theta_rows);
                constexpr double mean_scale  = 1.0 / static_cast<double>(theta_rows);

                radial_fields(radial_S_r, i)  = sum_J * theta_scale;
                radial_fields(radial_V_r, i)  = sum_JR * theta_scale * 2.0 * pi;
                radial_fields(radial_Kn, i)   = sum_gttdivJR * mean_scale;
                radial_fields(radial_Kn_r, i) = sum_gttdivJR_r * mean_scale;
                radial_fields(radial_Ln_r, i) = sum_JdivR * mean_scale;
            }
        }
    };
} // namespace geometry::detail

namespace geometry
{
    using detail::GeometryRuntime;
    using detail::radial_field_count;
    using detail::radial_Kn;
    using detail::radial_Kn_r;
    using detail::radial_Ln_r;
    using detail::radial_S_r;
    using detail::radial_V_r;
    using detail::surface_field_count;
    using detail::surface_grtdivJR_t;
    using detail::surface_gttdivJR;
    using detail::surface_gttdivJR_r;
    using detail::surface_J;
    using detail::surface_JdivR;
    using detail::surface_R;
    using detail::surface_R_t;
    using detail::surface_sin_tb;
    using detail::surface_Z_t;
} // namespace geometry
