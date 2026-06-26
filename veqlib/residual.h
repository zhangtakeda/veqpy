#pragma once

#include "geometry.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"
#include <cstddef>

namespace residual::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Tensor;
    using tensor::Vector;
    using tensor::uninitialized;

    inline constexpr size_t surface_G              = 0;
    inline constexpr size_t surface_Gpsin_R        = 1;
    inline constexpr size_t surface_Gpsin_Z        = 2;
    inline constexpr size_t surface_Gpsin_R_sin_tb = 3;
    inline constexpr size_t residual_surface_count = 4;

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

        // Store [rho][field][theta] to match the point-wise residual update
        // producer. The logical accessor remains [field, rho, theta].
        using SurfaceSlab  = Tensor<double, radial_nodes, residual_surface_count, theta_rows>;
        using RadialVector = Vector<double, radial_nodes>;
        using MomentRows   = Matrix<double, Shape::profile_count, radial_nodes>;
        using PackedVector = Vector<double, Shape::x_size>;

        SurfaceSlab  surface_fields{};
        RadialVector scratch{};

        constexpr void clear() noexcept
        {
            surface_fields.clear();
            scratch.clear();
        }

        constexpr double& surface_field(size_t row, size_t radial_node, size_t theta_node) noexcept
        {
            return surface_fields(radial_node, row, theta_node);
        }

        constexpr double surface_field(size_t row, size_t radial_node, size_t theta_node) const noexcept
        {
            return surface_fields(radial_node, row, theta_node);
        }

        template <typename SourceRuntime, typename GeometryRuntime>
        constexpr void update_compact(const SourceRuntime&   source_runtime,
                                      const GeometryRuntime& geometry_runtime) noexcept
        {
            static_assert(SourceRuntime::radial_nodes == radial_nodes, "residual/source radial grids must match");
            static_assert(GeometryRuntime::radial_nodes == radial_nodes, "residual/geometry radial grids must match");
            static_assert(GeometryRuntime::theta_rows == theta_rows, "residual/geometry theta grids must match");

            const double alpha1 = source_runtime.alpha1;
            const double alpha2 = source_runtime.alpha2;

            constexpr size_t geometry_row_stride    = theta_rows;
            constexpr size_t geometry_radial_stride = geometry::surface_field_count * theta_rows;
            constexpr size_t residual_row_stride    = theta_rows;
            constexpr size_t residual_radial_stride = residual_surface_count * theta_rows;

            const double* const geometry_surface = geometry_runtime.surface_fields.aligned_data();
            double* const       residual_surface = surface_fields.aligned_data();

            for (size_t i = 0; i < radial_nodes; ++i)
            {
                const double psin_r_i   = source_runtime.profile_root_fields(source::root_psin_r, i);
                const double psin_rr_i  = source_runtime.profile_root_fields(source::root_psin_rr, i);
                const double FFn_psin_i = source_runtime.FFn_psin[i];
                const double Pn_psin_i  = source_runtime.Pn_psin[i];
                const size_t geometry_radial_base = i * geometry_radial_stride;
                const size_t residual_radial_base = i * residual_radial_stride;

                for (size_t j = 0; j < theta_rows; ++j)
                {
                    const size_t geometry_base = geometry_radial_base + j;
                    const double sin_tb_ij =
                        geometry_surface[geometry_base + geometry::surface_sin_tb * geometry_row_stride];
                    const double R_ij = geometry_surface[geometry_base + geometry::surface_R * geometry_row_stride];
                    const double R_t_ij =
                        geometry_surface[geometry_base + geometry::surface_R_t * geometry_row_stride];
                    const double Z_t_ij =
                        geometry_surface[geometry_base + geometry::surface_Z_t * geometry_row_stride];
                    const double inv_J =
                        geometry_surface[geometry_base + geometry::surface_inv_J * geometry_row_stride];
                    const double JdivR_ij =
                        geometry_surface[geometry_base + geometry::surface_JdivR * geometry_row_stride];
                    const double grtdivJR_t_ij =
                        geometry_surface[geometry_base + geometry::surface_grtdivJR_t * geometry_row_stride];
                    const double gttdivJR_ij =
                        geometry_surface[geometry_base + geometry::surface_gttdivJR * geometry_row_stride];
                    const double gttdivJR_r_ij =
                        geometry_surface[geometry_base + geometry::surface_gttdivJR_r * geometry_row_stride];

                    const double psin_R = -Z_t_ij * inv_J * psin_r_i;
                    const double psin_Z = R_t_ij * inv_J * psin_r_i;

                    const double G1n  = JdivR_ij * (FFn_psin_i + R_ij * R_ij * Pn_psin_i);
                    const double G2n  = gttdivJR_ij * psin_rr_i + (gttdivJR_r_ij - grtdivJR_t_ij) * psin_r_i;
                    const double G_ij = alpha1 * G1n + alpha2 * G2n;
                    const size_t residual_base = residual_radial_base + j;
                    residual_surface[residual_base + surface_G * residual_row_stride] = G_ij;

                    const double Gpsin_R = G_ij * psin_R;
                    residual_surface[residual_base + surface_Gpsin_R * residual_row_stride]        = Gpsin_R;
                    residual_surface[residual_base + surface_Gpsin_Z * residual_row_stride]        = G_ij * psin_Z;
                    residual_surface[residual_base + surface_Gpsin_R_sin_tb * residual_row_stride] = Gpsin_R * sin_tb_ij;
                }
            }
        }

        constexpr PackedVector pack(double a, double R0, double B0) noexcept
        {
            PackedVector out{uninitialized};
            pack_profile<0>(out, a, R0, B0);
            return out;
        }

        constexpr void pack_into(PackedVector& out, double a, double R0, double B0) noexcept
        {
            pack_profile<0>(out, a, R0, B0);
        }

        constexpr void benchmark_theta_reduce_into(MomentRows& moments) noexcept
        {
            benchmark_theta_reduce_profile<0>(moments);
        }

        constexpr void benchmark_radial_project_from(
            PackedVector& out, const MomentRows& moments, double a, double R0, double B0) const noexcept
        {
            benchmark_radial_project_profile<0>(out, moments, a, R0, B0);
        }

    private:
        template <size_t ProfileId>
        constexpr void pack_profile(PackedVector& out, double a, double R0, double B0) noexcept
        {
            if constexpr (ProfileId < Shape::profile_count)
            {
                constexpr profiles::ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.optimized())
                    pack_one<ProfileId, slot.coefficient_count>(out, a, R0, B0);
                pack_profile<ProfileId + 1>(out, a, R0, B0);
            }
        }

        template <size_t ProfileId, size_t Count>
        constexpr void pack_one(PackedVector& out, double a, double R0, double B0) noexcept
        {
            constexpr size_t code         = block_code<ProfileId>();
            constexpr size_t order        = block_order<ProfileId>();
            constexpr size_t radial_power = block_radial_power<ProfileId>();

            if constexpr (code == block_h)
            {
                rowwise_sum(surface_Gpsin_R);
                project_scaled<Count, ProfileId>(out, GridType::y, unit_weights(), unit_weights(), a * base_scale());
            }
            else if constexpr (code == block_v)
            {
                rowwise_sum(surface_Gpsin_Z);
                project_scaled<Count, ProfileId>(out, GridType::y, unit_weights(), unit_weights(), a * base_scale());
            }
            else if constexpr (code == block_kappa)
            {
                rowwise_weighted_sum(surface_Gpsin_Z, theta_sin<1>());
                project_scaled<Count, ProfileId>(out, rho_power<1>(), GridType::y, unit_weights(), -a * base_scale());
            }
            else if constexpr (code == block_c0)
            {
                rowwise_sum(surface_Gpsin_R_sin_tb);
                project_scaled<Count, ProfileId>(out, rho_power<1>(), GridType::y, unit_weights(), -a * base_scale());
            }
            else if constexpr (code == block_c)
            {
                rowwise_weighted_sum(surface_Gpsin_R_sin_tb, theta_cos<order>());
                project_scaled<Count, ProfileId>(
                    out, rho_power<radial_power + 1>(), GridType::y, unit_weights(), -a * base_scale());
            }
            else if constexpr (code == block_s)
            {
                rowwise_weighted_sum(surface_Gpsin_R_sin_tb, theta_sin<order>());
                project_scaled<Count, ProfileId>(
                    out, rho_power<radial_power + 1>(), GridType::y, unit_weights(), -a * base_scale());
            }
            else if constexpr (code == block_psin)
            {
                rowwise_sum(surface_G);
                project_scaled<Count, ProfileId>(out, rho_power<2>(), GridType::y, unit_weights(), base_scale());
            }
            else if constexpr (code == block_F)
            {
                rowwise_sum(surface_G);
                project_scaled<Count, ProfileId>(
                    out, GridType::y, GridType::y, unit_weights(), base_scale() * (R0 * B0) * (R0 * B0));
            }
        }

        template <size_t ProfileId>
        constexpr void benchmark_theta_reduce_profile(MomentRows& moments) noexcept
        {
            if constexpr (ProfileId < Shape::profile_count)
            {
                constexpr profiles::ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.optimized())
                    benchmark_theta_reduce_one<ProfileId>(moments);
                benchmark_theta_reduce_profile<ProfileId + 1>(moments);
            }
        }

        template <size_t ProfileId>
        constexpr void benchmark_theta_reduce_one(MomentRows& moments) noexcept
        {
            constexpr size_t code  = block_code<ProfileId>();
            constexpr size_t order = block_order<ProfileId>();

            if constexpr (code == block_h)
                rowwise_sum_into(moments, ProfileId, surface_Gpsin_R);
            else if constexpr (code == block_v)
                rowwise_sum_into(moments, ProfileId, surface_Gpsin_Z);
            else if constexpr (code == block_kappa)
                rowwise_weighted_sum_into(moments, ProfileId, surface_Gpsin_Z, theta_sin<1>());
            else if constexpr (code == block_c0)
                rowwise_sum_into(moments, ProfileId, surface_Gpsin_R_sin_tb);
            else if constexpr (code == block_c)
                rowwise_weighted_sum_into(moments, ProfileId, surface_Gpsin_R_sin_tb, theta_cos<order>());
            else if constexpr (code == block_s)
                rowwise_weighted_sum_into(moments, ProfileId, surface_Gpsin_R_sin_tb, theta_sin<order>());
            else
                rowwise_sum_into(moments, ProfileId, surface_G);
        }

        template <size_t ProfileId>
        constexpr void benchmark_radial_project_profile(
            PackedVector& out, const MomentRows& moments, double a, double R0, double B0) const noexcept
        {
            if constexpr (ProfileId < Shape::profile_count)
            {
                constexpr profiles::ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.optimized())
                    benchmark_radial_project_one<ProfileId, slot.coefficient_count>(out, moments, a, R0, B0);
                benchmark_radial_project_profile<ProfileId + 1>(out, moments, a, R0, B0);
            }
        }

        template <size_t ProfileId, size_t Count>
        constexpr void benchmark_radial_project_one(
            PackedVector& out, const MomentRows& moments, double a, double R0, double B0) const noexcept
        {
            constexpr size_t code         = block_code<ProfileId>();
            constexpr size_t radial_power = block_radial_power<ProfileId>();

            if constexpr (code == block_h || code == block_v)
                project_moment_scaled<Count, ProfileId>(
                    out, moments, GridType::y, unit_weights(), unit_weights(), a * base_scale());
            else if constexpr (code == block_kappa || code == block_c0)
                project_moment_scaled<Count, ProfileId>(
                    out, moments, rho_power<1>(), GridType::y, unit_weights(), -a * base_scale());
            else if constexpr (code == block_c || code == block_s)
                project_moment_scaled<Count, ProfileId>(
                    out, moments, rho_power<radial_power + 1>(), GridType::y, unit_weights(), -a * base_scale());
            else if constexpr (code == block_psin)
                project_moment_scaled<Count, ProfileId>(
                    out, moments, rho_power<2>(), GridType::y, unit_weights(), base_scale());
            else if constexpr (code == block_F)
                project_moment_scaled<Count, ProfileId>(
                    out, moments, GridType::y, GridType::y, unit_weights(), base_scale() * (R0 * B0) * (R0 * B0));
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

        constexpr void rowwise_sum(size_t surface_row) noexcept
        {
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                double total = 0.0;
                for (size_t j = 0; j < theta_rows; ++j)
                    total += surface_field(surface_row, i, j);
                scratch[i] = total;
            }
        }

        template <typename ThetaWeights>
        constexpr void rowwise_weighted_sum(size_t surface_row, const ThetaWeights& theta_weights) noexcept
        {
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                double total = 0.0;
                for (size_t j = 0; j < theta_rows; ++j)
                    total += surface_field(surface_row, i, j) * theta_weights[j];
                scratch[i] = total;
            }
        }

        constexpr void rowwise_sum_into(MomentRows& moments, size_t profile_id, size_t surface_row) const noexcept
        {
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                double total = 0.0;
                for (size_t j = 0; j < theta_rows; ++j)
                    total += surface_field(surface_row, i, j);
                moments(profile_id, i) = total;
            }
        }

        template <typename ThetaWeights>
        constexpr void rowwise_weighted_sum_into(MomentRows&         moments,
                                                 size_t              profile_id,
                                                 size_t              surface_row,
                                                 const ThetaWeights& theta_weights) const noexcept
        {
            for (size_t i = 0; i < radial_nodes; ++i)
            {
                double total = 0.0;
                for (size_t j = 0; j < theta_rows; ++j)
                    total += surface_field(surface_row, i, j) * theta_weights[j];
                moments(profile_id, i) = total;
            }
        }

        template <size_t Count, size_t ProfileId, typename WeightA, typename WeightB, typename WeightC>
        constexpr void project_scaled(PackedVector&  out,
                                      const WeightA& weight_a,
                                      const WeightB& weight_b,
                                      const WeightC& weight_c,
                                      double         scalar) const noexcept
        {
            for (size_t degree = 0; degree < Count; ++degree)
            {
                double total = 0.0;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    total += projection_basis(degree, i) * scratch[i] * weight_a[i] * weight_b[i] * weight_c[i] *
                             GridType::weights[i];
                }
                out[static_cast<size_t>(Shape::coeff_index[ProfileId][degree])] = total * scalar;
            }
        }

        template <size_t Count, size_t ProfileId, typename WeightA, typename WeightB, typename WeightC>
        constexpr void project_moment_scaled(PackedVector&     out,
                                             const MomentRows& moments,
                                             const WeightA&    weight_a,
                                             const WeightB&    weight_b,
                                             const WeightC&    weight_c,
                                             double            scalar) const noexcept
        {
            for (size_t degree = 0; degree < Count; ++degree)
            {
                double total = 0.0;
                for (size_t i = 0; i < radial_nodes; ++i)
                {
                    total += projection_basis(degree, i) * moments(ProfileId, i) * weight_a[i] * weight_b[i] *
                             weight_c[i] * GridType::weights[i];
                }
                out[static_cast<size_t>(Shape::coeff_index[ProfileId][degree])] = total * scalar;
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

        template <size_t Order>
        static constexpr Vector<double, theta_rows> theta_sin() noexcept
        {
            Vector<double, theta_rows> out{uninitialized};
            for (size_t j = 0; j < theta_rows; ++j)
                out[j] = GridType::sin_mtheta(Order, j);
            return out;
        }

        template <size_t Order>
        static constexpr Vector<double, theta_rows> theta_cos() noexcept
        {
            Vector<double, theta_rows> out{uninitialized};
            for (size_t j = 0; j < theta_rows; ++j)
                out[j] = GridType::cos_mtheta(Order, j);
            return out;
        }
    };
} // namespace residual::detail

namespace residual
{
    using detail::ResidualRuntime;
    using detail::residual_surface_count;
    using detail::surface_G;
    using detail::surface_Gpsin_R;
    using detail::surface_Gpsin_R_sin_tb;
    using detail::surface_Gpsin_Z;
} // namespace residual
