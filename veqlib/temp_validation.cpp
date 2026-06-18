#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "grid.h"
#include "linalg.h"
#include "math.h"
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

    constexpr double tolerance = 1.0e-8;

    constexpr bool close(double lhs, double rhs, double tol = tolerance)
    {
        return math::abs(lhs - rhs) <= tol;
    }

    constexpr double pow_integer(double base, size_t exponent)
    {
        double value = 1.0;
        for (size_t i = 0; i < exponent; ++i)
            value *= base;
        return value;
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
                const double exact = degree == 0 ? 0.0
                                                 : static_cast<double>(degree) * pow_integer(nodes[row], degree - 1);
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
                const double exact =
                    pow_integer(nodes[row], degree + 1) / static_cast<double>(degree + 1);
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
    constexpr Matrix<double, 3, 4> thomas_band{0.0, -1.0, -1.0, -1.0,
                                               2.0, 2.0, 2.0, 2.0,
                                               -1.0, -1.0, -1.0, 0.0};
    constexpr Matrix<double, 4, 2> thomas_rhs{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0};

    constexpr bool linalg_constexpr_ok()
    {
        const auto product = matmul(dense_matrix, dense_matrix);
        if (!close(product[0], 10.0) || !close(product[1], 5.0) || !close(product[2], 5.0) ||
            !close(product[3], 5.0))
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
               close(cholesky[1], 3.0) && close(bunch[0], 2.0) && close(bunch[1], 3.0) &&
               close(qr[0], 2.0) && close(qr[1], 3.0) && close(doolittle_into[0], 2.0) &&
               close(doolittle_into[1], 3.0) && close(context_rhs[0], 2.0) &&
               close(context_rhs[1], 3.0) && close(thomas[0], 1.0) && close(thomas[1], 2.0) &&
               close(thomas[6], 1.0) && close(thomas[7], 2.0) && close(thomas_work[0], 1.0) &&
               close(thomas_work[1], 2.0) && close(thomas_work[6], 1.0) && close(thomas_work[7], 2.0);
    }

    constexpr bool tensor_math_constexpr_ok()
    {
        constexpr Vector<double, 3> values{1.0, 2.0, 3.0};
        constexpr auto              shifted = values + 1.0;
        constexpr auto              scaled  = 2.0 * values;
        constexpr auto              rooted  = math::sqrt(scaled + values);

        return close(math::sum(values), 6.0) && close(math::dot(values, values), 14.0) &&
               close(math::norm2(values), gcem::sqrt(14.0)) && close(shifted[2], 4.0) &&
               close(scaled[1], 4.0) && close(rooted[0], gcem::sqrt(3.0)) && math::is_finite(rooted);
    }

    constexpr bool grid_constexpr_ok()
    {
        using ProbeGrid = Grid<8, 16, Legendre, Spectral>;

        return ProbeGrid::nodes.count == 8 && ProbeGrid::weights.count == 8 &&
               ProbeGrid::accumulator.shape[0] == 8 && ProbeGrid::differentiator.shape[1] == 8 &&
               quadrature_shape_ok<Chebyshev, 8>() && quadrature_shape_ok<Legendre, 8>() &&
               quadrature_shape_ok<Lobatto, 8>() && quadrature_shape_ok<Radau, 8>() &&
               close(Lobatto::nodes<8>[0], 0.0, 0.0) && close(Lobatto::nodes<8>[7], 1.0, 0.0) &&
               close(Radau::nodes<8>[7], 1.0, 0.0) && max_moment_error<Chebyshev, 16>(7) < 1.0e-11 &&
               max_moment_error<Legendre, 16>(15) < 1.0e-10 &&
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

    static_assert(linalg_constexpr_ok());
    static_assert(tensor_math_constexpr_ok());
    static_assert(grid_constexpr_ok());

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
               close(root_x[0], 3.0, 1.0e-8) && lapack_info == 0 && close(lapack_b[0], 2.0) &&
               close(lapack_b[1], 3.0);
    }
}

int main()
{
    nlohmann::json report;

    report["constexpr"] = {
        {"linalg", linalg_constexpr_ok()},
        {"tensor_math", tensor_math_constexpr_ok()},
        {"grid", grid_constexpr_ok()},
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
                    runtime_library_ok(report);

    std::cout << report.dump(2) << '\n';
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
