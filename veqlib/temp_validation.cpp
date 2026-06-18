#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "grid.h"
#include "linalg.h"
#include "math.h"
#include "profiles.h"
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
    using linalg::BunchKaufman;
    using linalg::Cholesky;
    using linalg::Context;
    using linalg::Doolittle;
    using linalg::GolubReinsch;
    using linalg::Householder;
    using linalg::Thomas;
    using linalg::factorize;
    using linalg::factorize_into;
    using linalg::matmul;
    using linalg::matmul_into;
    using linalg::solve;
    using linalg::solve_into;
    using linalg::transpose;
    using linalg::transpose_into;
    using std::size_t;
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
    using ProbeProfiles = profiles::Profiles<
        Topology::L_max,
        Topology::K_max,
        Topology::h_count,
        Topology::v_count,
        Topology::kappa_count,
        Topology::psin_count,
        Topology::F_count,
        Topology::c_family_counts,
        Topology::s_family_counts>;

    static_assert(Topology::fourier_power<Topology::K_max + 7>() == Topology::K_max);
    static_assert(ProbeProfiles::fourier_power<Topology::K_max + 7>() == Topology::K_max);

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

    constexpr bool profiles_grid_constexpr_ok()
    {
        constexpr bool highest_c_ok = [] {
            if constexpr (Topology::C_max < ProbeProfiles::c_family_size)
                return c_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::C_max>();
            else
                return true;
        }();
        constexpr bool highest_s_ok = [] {
            if constexpr (Topology::S_max > 0)
                return s_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::S_max>();
            else
                return true;
        }();

        return h_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::h_count>() &&
               v_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::v_count>() &&
               kappa_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::kappa_count>() &&
               psin_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::psin_count>() &&
               F_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::F_count>() &&
               c_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() &&
               s_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() && highest_c_ok && highest_s_ok;
    }

    static_assert(linalg_constexpr_ok());
    static_assert(tensor_math_constexpr_ok());
    static_assert(grid_constexpr_ok());
    static_assert(profiles_grid_constexpr_ok());

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
                    profiles_grid_constexpr_ok() && runtime_library_ok(report);

    std::cout << report.dump(2) << '\n';
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
