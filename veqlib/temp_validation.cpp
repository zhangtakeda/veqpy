#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "geometry.h"
#include "grid.h"
#include "linalg.h"
#include "math.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"

namespace
{
    using grid::CFD33;
    using grid::CFD35;
    using grid::CFD55;
    using grid::Chebyshev;
    using grid::Grid;
    using grid::Legendre;
    using grid::Lobatto;
    using grid::Radau;
    using grid::Spectral;
    using geometry::GeometryRuntime;
    using geometry::radial_Kn;
    using geometry::radial_Ln_r;
    using geometry::radial_S_r;
    using geometry::radial_V_r;
    using geometry::surface_J;
    using geometry::surface_JdivR;
    using geometry::surface_R;
    using geometry::surface_R_t;
    using geometry::surface_sin_tb;
    using geometry::surface_Z_t;
    using linalg::BunchKaufman;
    using linalg::Cholesky;
    using linalg::Context;
    using linalg::Doolittle;
    using linalg::GolubReinsch;
    using linalg::Householder;
    using linalg::Thomas;
    using residual::ResidualRuntime;
    using linalg::factorize;
    using linalg::factorize_into;
    using linalg::matmul;
    using linalg::matmul_into;
    using linalg::solve;
    using linalg::solve_into;
    using linalg::transpose;
    using linalg::transpose_into;
    using std::size_t;
    using source::ProfileOwnedPsinSourceRuntime;
    using source::UniformSourceShape;
    using source::root_psin;
    using source::root_psin_r;
    using source::root_psin_rr;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    using Topology = config::DefaultTopology;
    using ProbeGrid = Grid<
        Topology::Nr,
        Topology::Nt,
        Topology::L_max,
        Topology::M_max,
        Topology::K_max,
        Legendre,
        Spectral>;
    using ProbeProfilesFromCounts = profiles::Profiles<
        Topology::L_max,
        Topology::K_max,
        Topology::h_count,
        Topology::v_count,
        Topology::kappa_count,
        Topology::psin_count,
        Topology::F_count,
        Topology::c_family_counts,
        Topology::s_family_counts>;

    constexpr auto topology_c_slots = profiles::tail_optimized_slots_from_counts<Topology::c_family_counts>();
    constexpr auto topology_s_slots = profiles::optimized_slots_from_counts<Topology::s_family_counts>();

    using ProbeProfileShape = profiles::ProfileShape<
        Topology::L_max,
        Topology::K_max,
        Topology::M_max,
        profiles::optimized_slot_from_count(Topology::h_count),
        profiles::optimized_slot_from_count(Topology::v_count),
        profiles::optimized_slot_from_count(Topology::kappa_count),
        profiles::first_optimized_slot_from_counts<Topology::c_family_counts>(),
        profiles::optimized_slot_from_count(Topology::psin_count),
        profiles::optimized_slot_from_count(Topology::F_count),
        topology_c_slots,
        topology_s_slots>;
    using ProbeProfiles = profiles::ProfileEvaluator<ProbeProfileShape>;

    constexpr auto no_c_slots = std::array<profiles::ProfileSlot, 0>{};
    constexpr auto no_s_slots = std::array<profiles::ProfileSlot, 0>{};

    using FixedOnlyProfileShape = profiles::ProfileShape<
        1,
        2,
        1,
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;

    using CircularGeometryShape = profiles::ProfileShape<
        2,
        2,
        2,
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using CircularGeometryGrid     = Grid<8, 8, 2, 2, 2, Legendre, Spectral>;
    using CircularGeometryProfiles = profiles::RuntimeProfiles<CircularGeometryShape, CircularGeometryGrid>;
    using CircularGeometryRuntime  = GeometryRuntime<CircularGeometryGrid>;

    using SourceMaterializationShape = profiles::ProfileShape<
        3,
        2,
        1,
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::optimized_slot(3),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using SourceMaterializationGrid     = Grid<8, 8, 3, 1, 2, Legendre, Spectral>;
    using SourceMaterializationProfiles = profiles::RuntimeProfiles<SourceMaterializationShape, SourceMaterializationGrid>;
    using SourceMaterializationRuntime =
        ProfileOwnedPsinSourceRuntime<SourceMaterializationGrid, UniformSourceShape<5>>;

    constexpr auto residual_c_slots = std::array{
        profiles::optimized_slot(2),
    };
    constexpr auto residual_s_slots = std::array{
        profiles::optimized_slot(2),
    };

    using ResidualProbeShape = profiles::ProfileShape<
        2,
        2,
        1,
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::absent_slot(),
        residual_c_slots,
        residual_s_slots>;
    using ResidualProbeGrid     = Grid<8, 8, 2, 1, 2, Legendre, Spectral>;
    using ResidualProbeProfiles = profiles::RuntimeProfiles<ResidualProbeShape, ResidualProbeGrid>;
    using ResidualProbeSource   = ProfileOwnedPsinSourceRuntime<ResidualProbeGrid, UniformSourceShape<5>>;
    using ResidualProbeGeometry = GeometryRuntime<ResidualProbeGrid>;
    using ResidualProbeRuntime  = ResidualRuntime<ResidualProbeShape, ResidualProbeGrid>;

    constexpr auto mixed_c_slots = std::array{
        profiles::optimized_slot(2),
        profiles::fixed_slot(),
    };
    constexpr auto mixed_s_slots = std::array{
        profiles::absent_slot(),
        profiles::optimized_slot(3),
    };

    using MixedProfileShape = profiles::ProfileShape<
        2,
        2,
        2,
        profiles::optimized_slot(2),
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        profiles::absent_slot(),
        mixed_c_slots,
        mixed_s_slots>;
    using RuntimeProbeGrid     = Grid<8, 8, 2, 2, 2, Legendre, Spectral>;
    using MixedRuntimeProfiles = profiles::RuntimeProfiles<MixedProfileShape, RuntimeProbeGrid>;

    constexpr auto semantic_c_slots = std::array{
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
    };
    constexpr auto semantic_s_slots = std::array{
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
    };

    using RuntimeSemanticShape = profiles::ProfileShape<
        2,
        2,
        4,
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
        semantic_c_slots,
        semantic_s_slots>;
    using RuntimeSemanticGrid     = Grid<8, 8, 2, 4, 2, Legendre, Spectral>;
    using RuntimeSemanticProfiles = profiles::RuntimeProfiles<RuntimeSemanticShape, RuntimeSemanticGrid>;
    using RuntimeSemanticEvaluator = profiles::ProfileEvaluator<RuntimeSemanticShape>;

    template <typename Shape, size_t HCount>
    consteval bool h_profile_L_matches()
    {
        if constexpr (HCount > 0)
            return Shape::profile_L[Shape::h_profile_id] == static_cast<int>(HCount - 1);
        else
            return Shape::profile_L[Shape::h_profile_id] == -1;
    }

    template <typename Shape, typename TopologyType, size_t SMax>
    consteval bool highest_s_profile_L_matches()
    {
        if constexpr (SMax > 0)
            return Shape::profile_L[Shape::template s_profile_id<SMax>()] ==
                   static_cast<int>(TopologyType::template s_count<SMax>() - 1);
        else
            return Shape::s_family_source_profile_ids[0] == -1;
    }

    static_assert(Topology::fourier_power<Topology::K_max + 7>() == Topology::K_max);
    static_assert(ProbeProfiles::fourier_power<Topology::K_max + 7>() == Topology::K_max);

    static_assert(ProbeProfileShape::h_profile_id == 0);
    static_assert(ProbeProfiles::shape::profile_count == ProbeProfilesFromCounts::shape::profile_count);
    static_assert(ProbeProfiles::shape::x_size == ProbeProfilesFromCounts::shape::x_size);
    static_assert(ProbeProfileShape::v_profile_id == 1);
    static_assert(ProbeProfileShape::kappa_profile_id == 2);
    static_assert(ProbeProfileShape::c_profile_id<0>() == 3);
    static_assert(ProbeProfileShape::s_profile_id<1>() == Topology::M_max + 4);
    static_assert(ProbeProfileShape::psin_profile_id == 2 * Topology::M_max + 4);
    static_assert(ProbeProfileShape::F_profile_id == 2 * Topology::M_max + 5);
    static_assert(ProbeProfileShape::profile_count == 2 * Topology::M_max + 6);
    static_assert(h_profile_L_matches<ProbeProfileShape, Topology::h_count>());
    static_assert(highest_s_profile_L_matches<ProbeProfileShape, Topology, Topology::S_max>());
    static_assert(ProbeProfileShape::coeff_index[ProbeProfileShape::h_profile_id][0] == 0);
    static_assert(ProbeProfileShape::order_offsets[0] == 0);
    static_assert(ProbeProfileShape::order_offsets[ProbeProfileShape::max_active_len] ==
                  static_cast<int>(ProbeProfileShape::x_size));
    static_assert(ProbeProfileShape::s_family_source_profile_ids[0] == -1);

    static_assert(FixedOnlyProfileShape::profile_count == 8);
    static_assert(FixedOnlyProfileShape::active_count == 0);
    static_assert(FixedOnlyProfileShape::max_active_len == 0);
    static_assert(FixedOnlyProfileShape::x_size == 0);
    static_assert(FixedOnlyProfileShape::profile_L[FixedOnlyProfileShape::h_profile_id] == -1);
    static_assert(FixedOnlyProfileShape::c_family_source_profile_ids[1] == -1);
    static_assert(FixedOnlyProfileShape::s_family_source_profile_ids[0] == -1);

    static_assert(MixedProfileShape::profile_count == 10);
    static_assert(MixedProfileShape::active_count == 4);
    static_assert(MixedProfileShape::max_active_len == 3);
    static_assert(MixedProfileShape::x_size == 8);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::h_profile_id] == 1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::v_profile_id] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<0>()] == 0);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<1>()] == 1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<2>()] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::s_profile_id<1>()] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::s_profile_id<2>()] == 2);
    static_assert(MixedProfileShape::active_profile_ids[0] == MixedProfileShape::h_profile_id);
    static_assert(MixedProfileShape::active_profile_ids[1] == MixedProfileShape::c_profile_id<0>());
    static_assert(MixedProfileShape::active_profile_ids[2] == MixedProfileShape::c_profile_id<1>());
    static_assert(MixedProfileShape::active_profile_ids[3] == MixedProfileShape::s_profile_id<2>());
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::h_profile_id][0] == 0);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<0>()][0] == 1);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<1>()][0] == 2);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][0] == 3);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::h_profile_id][1] == 4);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<1>()][1] == 5);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][1] == 6);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][2] == 7);
    static_assert(MixedProfileShape::order_offsets[0] == 0);
    static_assert(MixedProfileShape::order_offsets[1] == 4);
    static_assert(MixedProfileShape::order_offsets[2] == 7);
    static_assert(MixedProfileShape::order_offsets[3] == 8);
    static_assert(MixedProfileShape::c_family_source_profile_ids[0] ==
                  static_cast<int>(MixedProfileShape::c_profile_id<0>()));
    static_assert(MixedProfileShape::c_family_source_profile_ids[2] ==
                  static_cast<int>(MixedProfileShape::c_profile_id<2>()));
    static_assert(MixedProfileShape::s_family_source_profile_ids[0] == -1);
    static_assert(MixedProfileShape::s_family_source_profile_ids[1] == -1);
    static_assert(MixedProfileShape::s_family_source_profile_ids[2] ==
                  static_cast<int>(MixedProfileShape::s_profile_id<2>()));
    static_assert(MixedRuntimeProfiles::profile_field_count == MixedProfileShape::profile_count);
    static_assert(MixedRuntimeProfiles::family_field_count == MixedProfileShape::M_max + 1);
    static_assert(RuntimeSemanticEvaluator::fourier_power<4>() == RuntimeSemanticShape::K_max);

    constexpr double tolerance = 1.0e-8;

    constexpr bool close(double lhs, double rhs, double tol = tolerance) { return math::abs(lhs - rhs) <= tol; }

    constexpr double pow_integer(double base, size_t exponent)
    {
        double value = 1.0;
        for (size_t i = 0; i < exponent; ++i)
            value *= base;
        return value;
    }

    constexpr double power_value(double rho, size_t power) { return pow_integer(rho, power); }

    constexpr double power_radial(double rho, size_t power)
    {
        return power == 0 ? 0.0 : static_cast<double>(power) * pow_integer(rho, power - 1);
    }

    constexpr double power_radial2(double rho, size_t power)
    {
        return power < 2 ? 0.0 : static_cast<double>(power * (power - 1)) * pow_integer(rho, power - 2);
    }

    template <size_t Count>
    constexpr Vector<double, Count> make_profile_coefficients(double base, double step)
    {
        Vector<double, Count> coeffs{uninitialized};
        for (size_t i = 0; i < Count; ++i)
            coeffs[i] = base + step * static_cast<double>(i);
        return coeffs;
    }

    template <typename Shape, size_t ProfileId, size_t Count>
    constexpr void write_profile_coefficients(Vector<double, Shape::x_size>& x, const Vector<double, Count>& coeffs)
    {
        for (size_t degree = 0; degree < Count; ++degree)
            x[static_cast<size_t>(Shape::coeff_index[ProfileId][degree])] = coeffs[degree];
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_value(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = coeffs[0];
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_radial(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = 0.0;
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T_r(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_radial2(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = 0.0;
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T_rr(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr bool check_enveloped_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                           const Vector<double, Count>&                    coeffs,
                                           size_t                                         node)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);

        return close(profiles(node, 0), y * value) &&
               close(profiles(node, 1), -2.0 * rho * value + y * radial) &&
               close(profiles(node, 2), -2.0 * value - 4.0 * rho * radial + y * radial2);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_kappa_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                       const Vector<double, Count>&                    coeffs,
                                       size_t                                         node,
                                       double                                         ka)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;

        return close(profiles(node, 0), ka + base) &&
               close(profiles(node, 1), base_r) &&
               close(profiles(node, 2), base_rr);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_psin_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                      const Vector<double, Count>&                    coeffs,
                                      size_t                                         node)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp     = 1.0 + base;
        const double rp      = rho * rho;
        const double rp_r    = 2.0 * rho;
        const double rp_rr   = 2.0;

        return close(profiles(node, 0), rp * amp) &&
               close(profiles(node, 1), rp_r * amp + rp * base_r) &&
               close(profiles(node, 2), rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_F_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                   const Vector<double, Count>&                    coeffs,
                                   size_t                                         node,
                                   double                                         scale)
    {
        const double rho               = GridType::nodes[node];
        const double y                 = GridType::y[node];
        const double value             = profile_poly_value<GridType>(coeffs, node);
        const double radial            = profile_poly_radial<GridType>(coeffs, node);
        const double radial2           = profile_poly_radial2<GridType>(coeffs, node);
        const double base              = y * value;
        const double base_r            = -2.0 * rho * value + y * radial;
        const double base_rr           = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp_raw_unclamped = 1.0 + base;
        const double amp_raw           = math::max(amp_raw_unclamped, 1.0e-10);
        const double amp               = math::sqrt(amp_raw);
        const double inv_amp           = 1.0 / amp;
        const double inv_amp3          = inv_amp / amp_raw;
        const double amp_r             = 0.5 * base_r * inv_amp;
        const double amp_rr            = 0.5 * base_rr * inv_amp - 0.25 * base_r * base_r * inv_amp3;

        return close(profiles(node, 0), scale * amp) &&
               close(profiles(node, 1), scale * amp_r) &&
               close(profiles(node, 2), scale * amp_rr);
    }

    template <size_t Power, typename GridType, size_t Count>
    constexpr bool check_fourier_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                         const Vector<double, Count>&                    coeffs,
                                         size_t                                         node,
                                         double                                         offset)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp     = offset + base;
        const double rp      = power_value(rho, Power);
        const double rp_r    = power_radial(rho, Power);
        const double rp_rr   = power_radial2(rho, Power);

        return close(profiles(node, 0), rp * amp) &&
               close(profiles(node, 1), rp_r * amp + rp * base_r) &&
               close(profiles(node, 2), rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr);
    }

    template <typename Values>
    constexpr double sum_values(const Values& values)
    {
        double total = 0.0;
        for (size_t i = 0; i < Values::count; ++i)
            total += values[i];
        return total;
    }

    template <typename Quadrature, size_t N>
    constexpr double max_moment_error(size_t max_degree)
    {
        const auto& nodes   = Quadrature::template nodes<N>;
        const auto& weights = Quadrature::template weights<N>;
        double      worst   = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
        {
            double value = 0.0;
            for (size_t i = 0; i < N; ++i)
                value += weights[i] * pow_integer(nodes[i], degree);

            const double exact = 1.0 / static_cast<double>(degree + 1);
            const double error = math::abs(value - exact);
            if (error > worst)
                worst = error;
        }
        return worst;
    }

    template <typename Quadrature, size_t N>
    constexpr bool quadrature_shape_ok()
    {
        const auto& nodes   = Quadrature::template nodes<N>;
        const auto& weights = Quadrature::template weights<N>;
        if (!close(sum_values(weights), 1.0, 1.0e-12))
            return false;

        for (size_t i = 0; i < N; ++i)
        {
            if (!math::is_finite(nodes[i]) || !math::is_finite(weights[i]))
                return false;
            if (nodes[i] < 0.0 || nodes[i] > 1.0 || weights[i] <= 0.0)
                return false;
            if (i > 0 && nodes[i] <= nodes[i - 1])
                return false;
        }
        return true;
    }

    template <typename MatrixType, typename Nodes>
    constexpr double apply_to_power(const MatrixType& matrix, const Nodes& nodes, size_t row, size_t power)
    {
        const auto* values = matrix.data();
        double      total  = 0.0;
        for (size_t col = 0; col < Nodes::count; ++col)
            total += values[row * Nodes::count + col] * pow_integer(nodes[col], power);
        return total;
    }

    template <typename Calculus, typename Quadrature, size_t N>
    constexpr double max_differentiator_error(size_t max_degree)
    {
        const auto& nodes = Quadrature::template nodes<N>;
        const auto& diff  = Calculus::template differentiator<N, Quadrature>;
        double      worst = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
            for (size_t row = 0; row < N; ++row)
            {
                const double exact =
                    degree == 0 ? 0.0 : static_cast<double>(degree) * pow_integer(nodes[row], degree - 1);
                const double error = math::abs(apply_to_power(diff, nodes, row, degree) - exact);
                if (error > worst)
                    worst = error;
            }
        return worst;
    }

    template <typename Calculus, typename Quadrature, size_t N>
    constexpr double max_accumulator_error(size_t max_degree)
    {
        const auto& nodes = Quadrature::template nodes<N>;
        const auto& acc   = Calculus::template accumulator<N, Quadrature>;
        double      worst = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
            for (size_t row = 0; row < N; ++row)
            {
                const double exact = pow_integer(nodes[row], degree + 1) / static_cast<double>(degree + 1);
                const double error = math::abs(apply_to_power(acc, nodes, row, degree) - exact);
                if (error > worst)
                    worst = error;
            }
        return worst;
    }

    constexpr Matrix<double, 2, 2> dense_matrix{3.0, 1.0, 1.0, 2.0};
    constexpr Matrix<double, 2, 1> dense_rhs{9.0, 8.0};
    constexpr Matrix<double, 3, 2> tall_matrix{1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    constexpr Matrix<double, 3, 1> tall_rhs{2.0, 3.0, 5.0};
    constexpr Matrix<double, 3, 4> thomas_band{0.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0};
    constexpr Matrix<double, 4, 2> thomas_rhs{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0};

    constexpr bool linalg_constexpr_ok()
    {
        const auto product = matmul(dense_matrix, dense_matrix);
        if (!close(product[0], 10.0) || !close(product[1], 5.0) || !close(product[2], 5.0) || !close(product[3], 5.0))
            return false;

        Matrix<double, 2, 2> product_into{uninitialized};
        matmul_into(product_into, dense_matrix, dense_matrix);
        if (!close(product_into[0], product[0]) || !close(product_into[3], product[3]))
            return false;

        Matrix<double, 2, 2> transposed = transpose(dense_matrix);
        transpose_into(transposed, transposed);
        if (!close(transposed[1], dense_matrix[1]) || !close(transposed[2], dense_matrix[2]))
            return false;

        const auto doolittle = solve<Doolittle>(dense_matrix, dense_rhs);
        const auto cholesky  = solve<Cholesky>(dense_matrix, dense_rhs);
        const auto bunch     = solve<BunchKaufman>(dense_matrix, dense_rhs);
        const auto qr        = solve<Householder>(tall_matrix, tall_rhs);
        const auto thomas    = solve<Thomas>(thomas_band, thomas_rhs);

        Matrix<double, 2, 1> doolittle_into{uninitialized};
        solve_into<Doolittle>(doolittle_into, dense_matrix, dense_rhs);

        Context<Doolittle, 2, 2> context;
        factorize_into<Doolittle>(context, dense_matrix);
        auto context_rhs = dense_rhs;
        context.substitute_inplace<1>(context_rhs.data());

        const auto thomas_context = factorize<Thomas>(thomas_band);
        auto       thomas_work    = thomas_rhs;
        thomas_context.substitute_inplace<2>(thomas_work.data());

        return close(doolittle[0], 2.0) && close(doolittle[1], 3.0) && close(cholesky[0], 2.0) &&
               close(cholesky[1], 3.0) && close(bunch[0], 2.0) && close(bunch[1], 3.0) && close(qr[0], 2.0) &&
               close(qr[1], 3.0) && close(doolittle_into[0], 2.0) && close(doolittle_into[1], 3.0) &&
               close(context_rhs[0], 2.0) && close(context_rhs[1], 3.0) && close(thomas[0], 1.0) &&
               close(thomas[1], 2.0) && close(thomas[6], 1.0) && close(thomas[7], 2.0) && close(thomas_work[0], 1.0) &&
               close(thomas_work[1], 2.0) && close(thomas_work[6], 1.0) && close(thomas_work[7], 2.0);
    }

    constexpr bool tensor_math_constexpr_ok()
    {
        constexpr Vector<double, 3> values{1.0, 2.0, 3.0};
        constexpr auto              shifted = values + 1.0;
        constexpr auto              scaled  = 2.0 * values;
        constexpr auto              rooted  = math::sqrt(scaled + values);

        return close(math::sum(values), 6.0) && close(math::dot(values, values), 14.0) &&
               close(math::norm2(values), gcem::sqrt(14.0)) && close(shifted[2], 4.0) && close(scaled[1], 4.0) &&
               close(rooted[0], gcem::sqrt(3.0)) && math::is_finite(rooted);
    }

    constexpr bool grid_constexpr_ok()
    {
        constexpr double rho0              = ProbeGrid::nodes[0];
        constexpr double x0                = 2.0 * rho0 * rho0 - 1.0;
        constexpr double theta_step        = 2.0 * grid::detail::pi / static_cast<double>(ProbeGrid::theta_rows);
        constexpr bool   radial_tables_ok  = close(ProbeGrid::x[0], x0) &&
                                            close(ProbeGrid::y[0], 1.0 - rho0 * rho0) &&
                                            close(ProbeGrid::rhos(0, 0), rho0) &&
                                            close(ProbeGrid::rhos(1, 0), rho0 * rho0);
        constexpr bool theta_tables_ok = close(ProbeGrid::theta[0], 0.0, 0.0) &&
                                         close(ProbeGrid::theta[1], theta_step) &&
                                         close(ProbeGrid::cos_mtheta(0, 3), 1.0) &&
                                         close(ProbeGrid::sin_mtheta(0, 3), 0.0, 1.0e-15) &&
                                         close(ProbeGrid::m_cos_mtheta(0, 3), 0.0, 0.0) &&
                                         close(ProbeGrid::m2_sin_mtheta(0, 3), 0.0, 0.0);
        constexpr bool chebyshev_tables_ok = close(ProbeGrid::T(0, 0), ProbeGrid::x[0]) &&
                                             close(ProbeGrid::T_r(0, 0), 4.0 * ProbeGrid::nodes[0]) &&
                                             close(ProbeGrid::T_rr(0, 0), 4.0);
        constexpr bool harmonic_tables_ok = close(ProbeGrid::cos_mtheta(1, 2), math::cos(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::sin_mtheta(1, 2), math::sin(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::m_cos_mtheta(1, 2), math::cos(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::m_sin_mtheta(1, 2), math::sin(ProbeGrid::theta[2]));

        return ProbeGrid::nodes.count == ProbeGrid::radial_nodes &&
               ProbeGrid::weights.count == ProbeGrid::radial_nodes &&
               ProbeGrid::accumulator.shape[0] == ProbeGrid::radial_nodes &&
               ProbeGrid::differentiator.shape[1] == ProbeGrid::radial_nodes && radial_tables_ok && theta_tables_ok &&
               chebyshev_tables_ok && harmonic_tables_ok && quadrature_shape_ok<Chebyshev, 8>() &&
               quadrature_shape_ok<Legendre, 8>() && quadrature_shape_ok<Lobatto, 8>() &&
               quadrature_shape_ok<Radau, 8>() && close(Lobatto::nodes<8>[0], 0.0, 0.0) &&
               close(Lobatto::nodes<8>[7], 1.0, 0.0) && close(Radau::nodes<8>[7], 1.0, 0.0) &&
               max_moment_error<Chebyshev, 16>(7) < 1.0e-11 && max_moment_error<Legendre, 16>(15) < 1.0e-10 &&
               max_moment_error<Lobatto, 16>(13) < 1.0e-10 && max_moment_error<Radau, 16>(14) < 1.0e-10 &&
               max_differentiator_error<Spectral, Chebyshev, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Legendre, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Lobatto, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Radau, 8>(3) < 1.0e-8 &&
               max_accumulator_error<Spectral, Chebyshev, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Legendre, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Lobatto, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Radau, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD33, Lobatto, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD35, Lobatto, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD55, Lobatto, 8>(2) < 1.0e-8 &&
               max_accumulator_error<CFD33, Lobatto, 8>(1) < 1.0e-8 &&
               max_accumulator_error<CFD35, Lobatto, 8>(1) < 1.0e-8 &&
               max_accumulator_error<CFD55, Lobatto, 8>(1) < 1.0e-8;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool h_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(0.12, -0.003);
            ProfileOps::update_h(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_enveloped_profile<ProbeGrid>(out, coeffs, 0) &&
                   check_enveloped_profile<ProbeGrid>(out, coeffs, ProbeGrid::nodes.count - 1);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool v_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(-0.08, 0.002);
            ProfileOps::update_v(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_enveloped_profile<ProbeGrid>(out, coeffs, 1);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool kappa_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto   coeffs = make_profile_coefficients<Count>(0.05, 0.001);
            constexpr double ka = 1.7;
            ProfileOps::update_kappa(
                out,
                coeffs,
                ProbeGrid::T,
                ProbeGrid::T_r,
                ProbeGrid::T_rr,
                ProbeGrid::rhos,
                ka
            );
            return math::is_finite(out) && check_kappa_profile<ProbeGrid>(out, coeffs, 2, ka);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool psin_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(0.02, -0.0005);
            ProfileOps::update_psin(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_psin_profile<ProbeGrid>(out, coeffs, 3);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool F_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto   coeffs = make_profile_coefficients<Count>(0.015, -0.0004);
            constexpr double scale = 2.25;
            ProfileOps::update_F(
                out,
                coeffs,
                ProbeGrid::T,
                ProbeGrid::T_r,
                ProbeGrid::T_rr,
                ProbeGrid::rhos,
                scale
            );
            return math::is_finite(out) && check_F_profile<ProbeGrid>(out, coeffs, 4, scale);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool c_profile_grid_ok()
    {
        if constexpr (Order < ProfileOps::c_family_size)
        {
            constexpr size_t count = ProfileOps::template c_count<Order>();
            if constexpr (count > 0)
            {
                constexpr size_t power = ProfileOps::template fourier_power<Order>();

                Matrix<double, ProbeGrid::nodes.count, 3> out{};
                const auto   coeffs = make_profile_coefficients<count>(0.07, 0.001);
                constexpr double offset = 0.25;
                ProfileOps::template update_c<Order>(
                    out,
                    coeffs,
                    ProbeGrid::T,
                    ProbeGrid::T_r,
                    ProbeGrid::T_rr,
                    ProbeGrid::rhos,
                    offset
                );
                return math::is_finite(out) && check_fourier_profile<power, ProbeGrid>(out, coeffs, 5, offset);
            }
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool s_profile_grid_ok()
    {
        static_assert(Order > 0, "s profile checks start at s1");

        if constexpr (Order <= ProfileOps::s_family_size)
        {
            constexpr size_t count = ProfileOps::template s_count<Order>();
            if constexpr (count > 0)
            {
                constexpr size_t power = ProfileOps::template fourier_power<Order>();

                Matrix<double, ProbeGrid::nodes.count, 3> out{};
                const auto   coeffs = make_profile_coefficients<count>(-0.06, 0.0015);
                constexpr double offset = -0.15;
                ProfileOps::template update_s<Order>(
                    out,
                    coeffs,
                    ProbeGrid::T,
                    ProbeGrid::T_r,
                    ProbeGrid::T_rr,
                    ProbeGrid::rhos,
                    offset
                );
                return math::is_finite(out) && check_fourier_profile<power, ProbeGrid>(out, coeffs, 6, offset);
            }
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool highest_s_profile_grid_ok()
    {
        if constexpr (Order > 0)
            return s_profile_grid_ok<ProfileOps, ProbeGrid, Order>();
        else
            return true;
    }

    constexpr bool profiles_grid_constexpr_ok()
    {
        constexpr bool highest_c_ok = [] {
            if constexpr (Topology::C_max < ProbeProfiles::c_family_size)
                return c_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::C_max>();
            else
                return true;
        }();
        constexpr bool highest_s_ok = highest_s_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::S_max>();

        return h_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::h_count>() &&
               v_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::v_count>() &&
               kappa_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::kappa_count>() &&
               psin_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::psin_count>() &&
               F_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::F_count>() &&
               c_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() &&
               s_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() && highest_c_ok && highest_s_ok;
    }

    constexpr bool runtime_profiles_constexpr_ok()
    {
        using Shape   = MixedProfileShape;
        using Runtime = MixedRuntimeProfiles;

        constexpr size_t h_id  = Shape::h_profile_id;
        constexpr size_t c0_id = Shape::c_profile_id<0>();
        constexpr size_t c1_id = Shape::c_profile_id<1>();
        constexpr size_t c2_id = Shape::c_profile_id<2>();
        constexpr size_t s2_id = Shape::s_profile_id<2>();

        const auto h_coeffs  = make_profile_coefficients<2>(0.12, -0.003);
        const auto c0_coeffs = make_profile_coefficients<1>(0.07, 0.001);
        const auto c1_coeffs = make_profile_coefficients<2>(0.08, 0.001);
        const auto s2_coeffs = make_profile_coefficients<3>(-0.06, 0.0015);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[c0_id] = 0.25;
        params.offsets[c1_id] = 0.35;
        params.offsets[c2_id] = 4.5;
        params.offsets[s2_id] = -0.15;
        params.scales[c2_id]  = 2.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, h_id>(x, h_coeffs);
        write_profile_coefficients<Shape, c0_id>(x, c0_coeffs);
        write_profile_coefficients<Shape, c1_id>(x, c1_coeffs);
        write_profile_coefficients<Shape, s2_id>(x, s2_coeffs);

        Runtime runtime{};
        runtime.refresh_fixed(params);
        runtime.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        return check_enveloped_profile<RuntimeProbeGrid>(runtime.template profile_matrix<h_id>(), h_coeffs, 0) &&
               check_fourier_profile<0, RuntimeProbeGrid>(
                   runtime.template profile_matrix<c0_id>(),
                   c0_coeffs,
                   5,
                   params.offsets[c0_id]
               ) &&
               check_fourier_profile<1, RuntimeProbeGrid>(
                   runtime.template profile_matrix<c1_id>(),
                   c1_coeffs,
                   5,
                   params.offsets[c1_id]
               ) &&
               check_fourier_profile<2, RuntimeProbeGrid>(
                   runtime.template profile_matrix<s2_id>(),
                   s2_coeffs,
                   6,
                   params.offsets[s2_id]
               ) &&
               close(runtime.template profile_field<c2_id>(0, 0), 9.0) &&
               close(runtime.template profile_field<c2_id>(0, 1), 0.0) &&
               close(runtime.template profile_field<c2_id>(0, 2), 0.0) &&
               close(runtime.template c_family_field<2>(0, 0), 9.0) &&
               close(runtime.template s_family_field<1>(0, 0), 0.0) &&
               close(runtime.template s_family_field<2>(0, 0), runtime.template profile_field<s2_id>(0, 0));
    }

    constexpr bool runtime_profile_semantics_constexpr_ok()
    {
        using Shape   = RuntimeSemanticShape;
        using Runtime = RuntimeSemanticProfiles;

        constexpr size_t c0_id = Shape::c_profile_id<0>();
        constexpr size_t c4_id = Shape::c_profile_id<4>();
        constexpr size_t s4_id = Shape::s_profile_id<4>();
        constexpr size_t F_id  = Shape::F_profile_id;

        const auto c0_coeffs = make_profile_coefficients<1>(0.04, 0.0);
        const auto c4_coeffs = make_profile_coefficients<2>(0.05, 0.002);
        const auto s4_coeffs = make_profile_coefficients<2>(-0.03, 0.001);
        const auto F_coeffs  = make_profile_coefficients<2>(0.015, -0.0004);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[c0_id] = 0.2;
        params.offsets[c4_id] = 0.3;
        params.offsets[s4_id] = -0.1;
        params.scales[F_id]   = 2.25;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, c0_id>(x, c0_coeffs);
        write_profile_coefficients<Shape, c4_id>(x, c4_coeffs);
        write_profile_coefficients<Shape, s4_id>(x, s4_coeffs);
        write_profile_coefficients<Shape, F_id>(x, F_coeffs);

        Runtime runtime{};
        runtime.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        return check_fourier_profile<0, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<c0_id>(),
                   c0_coeffs,
                   0,
                   params.offsets[c0_id]
               ) &&
               check_fourier_profile<2, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<c4_id>(),
                   c4_coeffs,
                   1,
                   params.offsets[c4_id]
               ) &&
               check_fourier_profile<2, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<s4_id>(),
                   s4_coeffs,
                   2,
                   params.offsets[s4_id]
               ) &&
               check_F_profile<RuntimeSemanticGrid>(
                   runtime.template profile_matrix<F_id>(),
                   F_coeffs,
                   3,
                   params.scales[F_id]
               ) &&
               close(runtime.template c_family_field<4>(0, 0), runtime.template profile_field<c4_id>(0, 0)) &&
               close(runtime.template s_family_field<4>(0, 0), runtime.template profile_field<s4_id>(0, 0));
    }

    constexpr bool geometry_circular_constexpr_ok()
    {
        using Shape   = CircularGeometryShape;
        using Grid    = CircularGeometryGrid;
        using Runtime = CircularGeometryProfiles;
        using Geometry = CircularGeometryRuntime;

        constexpr double a  = 0.42;
        constexpr double R0 = 1.8;
        constexpr double Z0 = -0.25;
        constexpr double ka = 1.55;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::kappa_profile_id] = ka;
        params.scales[Shape::kappa_profile_id]  = 1.0;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;
        params.scales[Shape::c_profile_id<0>()]  = 1.0;

        Runtime profiles{};
        profiles.refresh_fixed(params);

        Geometry geometry{};
        geometry.update(a, R0, Z0, profiles);

        if (!math::is_finite(geometry.surface_fields) || !math::is_finite(geometry.radial_fields))
            return false;

        for (size_t i = 0; i < Grid::radial_nodes; ++i)
        {
            const double rho_i      = Grid::nodes[i];
            const double expected_J = a * a * rho_i * ka;
            const double expected_S = 2.0 * grid::detail::pi * expected_J;
            const double expected_V = 4.0 * grid::detail::pi * grid::detail::pi * expected_J * R0;

            if (!close(geometry.radial_field(radial_S_r, i), expected_S, 1.0e-11) ||
                !close(geometry.radial_field(radial_V_r, i), expected_V, 1.0e-10))
                return false;
            if (geometry.radial_field(radial_Kn, i) <= 0.0 || geometry.radial_field(radial_Ln_r, i) <= 0.0)
                return false;

            for (size_t j = 0; j < Grid::theta_rows; ++j)
            {
                const double sin_t      = Grid::sin_mtheta(1, j);
                const double cos_t      = Grid::cos_mtheta(1, j);
                const double expected_R = R0 + a * rho_i * cos_t;

                if (!close(geometry.surface_field(surface_sin_tb, i, j), sin_t, 1.0e-12) ||
                    !close(geometry.surface_field(surface_R, i, j), expected_R, 1.0e-12) ||
                    !close(geometry.surface_field(surface_R_t, i, j), -a * rho_i * sin_t, 1.0e-12) ||
                    !close(geometry.surface_field(surface_Z_t, i, j), -a * rho_i * ka * cos_t, 1.0e-12) ||
                    !close(geometry.surface_field(surface_J, i, j), expected_J, 1.0e-12) ||
                    !close(geometry.surface_field(surface_JdivR, i, j), expected_J / expected_R, 1.0e-12))
                    return false;
            }
        }

        return true;
    }

    constexpr bool source_materialization_constexpr_ok()
    {
        using Shape   = SourceMaterializationShape;
        using Grid    = SourceMaterializationGrid;
        using Runtime = SourceMaterializationProfiles;
        using Source  = SourceMaterializationRuntime;

        constexpr size_t psin_id = Shape::psin_profile_id;
        const auto       psin_coeffs = make_profile_coefficients<3>(0.01, -0.0002);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::h_profile_id] = 0.0;
        params.offsets[Shape::v_profile_id] = 0.0;
        params.offsets[Shape::kappa_profile_id] = 1.45;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, psin_id>(x, psin_coeffs);

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Source source{};
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{-1.0, -0.875, -0.75, -0.625, -0.5};
        source.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );

        if (!source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0)))
            return false;

        if (!close(source.source_target_root_fields(root_psin, 0), 0.0, 0.0) ||
            !close(source.source_target_root_fields(root_psin, Grid::radial_nodes - 1), 1.0, 0.0))
            return false;

        for (size_t i = 0; i < Grid::radial_nodes; ++i)
        {
            const double q = source.source_psin_query[i];
            if (!close(source.source_parameter_query[i], q) ||
                !close(source.source_target_root_fields(root_psin, i), q) ||
                source.source_target_root_fields(root_psin_r, i) <= 0.0 ||
                !math::is_finite(source.source_target_root_fields(root_psin_rr, i)))
                return false;

            const double expected_heat = 2.0 + 3.0 * q;
            const double expected_current = -1.0 + 0.5 * q;
            if (!close(source.materialized_heat_input[i], expected_heat, 1.0e-10) ||
                !close(source.materialized_current_input[i], expected_current, 1.0e-10))
                return false;
        }

        return true;
    }

    constexpr bool pf_source_constexpr_ok()
    {
        using Shape   = SourceMaterializationShape;
        using Grid    = SourceMaterializationGrid;
        using Runtime = SourceMaterializationProfiles;
        using Source  = SourceMaterializationRuntime;
        using Geometry = GeometryRuntime<Grid>;

        constexpr size_t psin_id = Shape::psin_profile_id;
        const auto       psin_coeffs = make_profile_coefficients<3>(0.01, -0.0002);
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        constexpr double a  = 0.42;
        constexpr double R0 = 1.8;
        constexpr double Z0 = -0.25;
        constexpr double B0 = 2.1;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::h_profile_id] = 0.0;
        params.offsets[Shape::v_profile_id] = 0.0;
        params.offsets[Shape::kappa_profile_id] = 1.45;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, psin_id>(x, psin_coeffs);

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Geometry geometry{};
        geometry.update(a, R0, Z0, profiles);

        auto make_source = [&heat, &current, &profiles](Source& source) constexpr {
            source.set_uniform_sources(
                std::span<const double, heat.size()>{heat.data(), heat.size()},
                std::span<const double, current.size()>{current.data(), current.size()}
            );
            return source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0));
        };

        Source free_source{};
        if (!make_source(free_source) ||
            !free_source.update_pf_from_psin_uniform(
                geometry,
                B0,
                source::unset_constraint(),
                source::unset_constraint(),
                source::axis_fix_count<Grid>(0.0)
            ))
            return false;

        if (!math::is_finite(free_source.alpha1) || !math::is_finite(free_source.alpha2))
            return false;
        for (size_t i = 0; i < Grid::radial_nodes; ++i)
        {
            if (free_source.source_target_root_fields(root_psin_r, i) <= 0.0 ||
                !close(free_source.Pn_psin[i], free_source.materialized_heat_input[i] / free_source.alpha1, 1.0e-10) ||
                !close(free_source.FFn_psin[i], free_source.materialized_current_input[i] / free_source.alpha1, 1.0e-10))
                return false;
        }

        Source ip_source{};
        if (!make_source(ip_source) ||
            !ip_source.update_pf_from_psin_uniform(
                geometry,
                B0,
                0.75,
                source::unset_constraint(),
                source::axis_fix_count<Grid>(0.0)
            ))
            return false;
        for (size_t i = 0; i < Grid::radial_nodes; ++i)
            if (!close(ip_source.Pn_psin[i], ip_source.materialized_heat_input[i]) ||
                !close(ip_source.FFn_psin[i], ip_source.materialized_current_input[i]))
                return false;

        Source beta_source{};
        if (!make_source(beta_source) ||
            !beta_source.update_pf_from_psin_uniform(
                geometry,
                B0,
                source::unset_constraint(),
                0.04,
                source::axis_fix_count<Grid>(0.0)
            ))
            return false;
        for (size_t i = 0; i < Grid::radial_nodes; ++i)
            if (!close(beta_source.Pn_psin[i], beta_source.materialized_heat_input[i]) ||
                !close(beta_source.FFn_psin[i], beta_source.materialized_current_input[i]))
                return false;

        Source invalid_source{};
        if (!make_source(invalid_source))
            return false;
        if (invalid_source.update_pf_from_psin_uniform(geometry, B0, 0.75, 0.04, source::axis_fix_count<Grid>(0.0)))
            return false;

        return math::is_finite(ip_source.alpha1) && math::is_finite(ip_source.alpha2) &&
               math::is_finite(beta_source.alpha1) && math::is_finite(beta_source.alpha2);
    }

    constexpr bool residual_pack_constexpr_ok()
    {
        using Shape    = ResidualProbeShape;
        using Grid     = ResidualProbeGrid;
        using Runtime  = ResidualProbeProfiles;
        using Source   = ResidualProbeSource;
        using Geometry = ResidualProbeGeometry;
        using Residual = ResidualProbeRuntime;

        constexpr size_t h_id     = Shape::h_profile_id;
        constexpr size_t v_id     = Shape::v_profile_id;
        constexpr size_t k_id     = Shape::kappa_profile_id;
        constexpr size_t c0_id    = Shape::c_profile_id<0>();
        constexpr size_t c1_id    = Shape::c_profile_id<1>();
        constexpr size_t s1_id    = Shape::s_profile_id<1>();
        constexpr size_t psin_id  = Shape::psin_profile_id;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[k_id]  = 1.45;
        params.offsets[c0_id] = 0.0;
        params.offsets[c1_id] = 0.0;
        params.offsets[s1_id] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, h_id>(x, make_profile_coefficients<2>(0.020, -0.0010));
        write_profile_coefficients<Shape, v_id>(x, make_profile_coefficients<2>(0.010, 0.0010));
        write_profile_coefficients<Shape, k_id>(x, make_profile_coefficients<2>(0.015, -0.0007));
        write_profile_coefficients<Shape, c0_id>(x, make_profile_coefficients<2>(0.004, 0.0002));
        write_profile_coefficients<Shape, c1_id>(x, make_profile_coefficients<2>(0.003, 0.0002));
        write_profile_coefficients<Shape, s1_id>(x, make_profile_coefficients<2>(-0.002, 0.0001));
        write_profile_coefficients<Shape, psin_id>(x, make_profile_coefficients<2>(0.010, -0.0002));

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Geometry geometry{};
        geometry.update(0.42, 1.8, -0.25, profiles);

        Source source{};
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        source.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );
        if (!source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0)) ||
            !source.update_pf_from_psin_uniform(
                geometry,
                2.1,
                source::unset_constraint(),
                source::unset_constraint(),
                source::axis_fix_count<Grid>(0.0)
            ))
            return false;

        Residual residual{};
        residual.update_compact(source, geometry);
        const auto packed = residual.pack(0.42, 1.8, 2.1);

        if (!math::is_finite(residual.surface_fields) || !math::is_finite(packed))
            return false;

        double norm1 = 0.0;
        for (size_t i = 0; i < Shape::x_size; ++i)
            norm1 += math::abs(packed[i]);
        return norm1 > 1.0e-12;
    }

    static_assert(linalg_constexpr_ok());
    static_assert(tensor_math_constexpr_ok());
    static_assert(grid_constexpr_ok());
    static_assert(profiles_grid_constexpr_ok());
    static_assert(runtime_profiles_constexpr_ok());
    static_assert(runtime_profile_semantics_constexpr_ok());
    static_assert(geometry_circular_constexpr_ok());
    static_assert(source_materialization_constexpr_ok());
    static_assert(pf_source_constexpr_ok());
    static_assert(residual_pack_constexpr_ok());

    int root_residual(void*, int n, const double* x, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != 1)
            return 0;
        fvec[0] = x[0] * x[0] - 9.0;
        return 0;
    }

    bool runtime_library_ok(nlohmann::json& report)
    {
        const auto svd_solution = solve<GolubReinsch>(dense_matrix, dense_rhs);

        double        root_x[1] = {4.0};
        double        root_f[1] = {0.0};
        constexpr int root_n    = 1;
        constexpr int root_lwa  = root_n * (3 * root_n + 13) / 2;
        double        root_work[root_lwa];
        const int     root_info = hybrd1(root_residual, nullptr, root_n, root_x, root_f, 1.0e-10, root_work, root_lwa);

        double           lapack_a[4] = {3.0, 1.0, 1.0, 2.0};
        double           lapack_b[2] = {9.0, 8.0};
        lapack_int       ipiv[2];
        const lapack_int lapack_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, lapack_a, 2, ipiv, lapack_b, 1);

        report["runtime"] = {
            {"gcem_sqrt_25", gcem::sqrt(25.0)},
            {"golub_reinsch", {svd_solution[0], svd_solution[1]}},
            {"cminpack", {{"info", root_info}, {"x", root_x[0]}, {"f", root_f[0]}}},
            {"lapacke", {{"info", static_cast<int>(lapack_info)}, {"solution", {lapack_b[0], lapack_b[1]}}}},
        };

        return close(svd_solution[0], 2.0) && close(svd_solution[1], 3.0) && root_info > 0 &&
               close(root_x[0], 3.0, 1.0e-8) && lapack_info == 0 && close(lapack_b[0], 2.0) && close(lapack_b[1], 3.0);
    }
} // namespace

int main()
{
    nlohmann::json report;

    report["constexpr"] = {
        {"linalg", linalg_constexpr_ok()},
        {"tensor_math", tensor_math_constexpr_ok()},
        {"grid", grid_constexpr_ok()},
        {"profiles_grid", profiles_grid_constexpr_ok()},
        {"runtime_profiles", runtime_profiles_constexpr_ok()},
        {"runtime_profile_semantics", runtime_profile_semantics_constexpr_ok()},
        {"geometry_circular", geometry_circular_constexpr_ok()},
        {"source_materialization", source_materialization_constexpr_ok()},
        {"pf_source", pf_source_constexpr_ok()},
        {"residual_pack", residual_pack_constexpr_ok()},
    };
    report["quadrature"] = {
        {"chebyshev_moment_error_n16_degree7", max_moment_error<Chebyshev, 16>(7)},
        {"legendre_moment_error_n16_degree15", max_moment_error<Legendre, 16>(15)},
        {"lobatto_moment_error_n16_degree13", max_moment_error<Lobatto, 16>(13)},
        {"radau_moment_error_n16_degree14", max_moment_error<Radau, 16>(14)},
    };
    report["calculus"] = {
        {"spectral_legendre_diff_error", max_differentiator_error<Spectral, Legendre, 8>(3)},
        {"spectral_legendre_acc_error", max_accumulator_error<Spectral, Legendre, 8>(2)},
        {"cfd33_lobatto_diff_error", max_differentiator_error<CFD33, Lobatto, 8>(2)},
        {"cfd35_lobatto_diff_error", max_differentiator_error<CFD35, Lobatto, 8>(2)},
        {"cfd55_lobatto_diff_error", max_differentiator_error<CFD55, Lobatto, 8>(2)},
        {"cfd33_lobatto_acc_error", max_accumulator_error<CFD33, Lobatto, 8>(1)},
        {"cfd35_lobatto_acc_error", max_accumulator_error<CFD35, Lobatto, 8>(1)},
        {"cfd55_lobatto_acc_error", max_accumulator_error<CFD55, Lobatto, 8>(1)},
    };

    const bool ok = linalg_constexpr_ok() && tensor_math_constexpr_ok() && grid_constexpr_ok() &&
                    profiles_grid_constexpr_ok() && runtime_profiles_constexpr_ok() &&
                    runtime_profile_semantics_constexpr_ok() && geometry_circular_constexpr_ok() &&
                    source_materialization_constexpr_ok() && pf_source_constexpr_ok() &&
                    residual_pack_constexpr_ok() && runtime_library_ok(report);

    std::cout << report.dump(2) << '\n';
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
