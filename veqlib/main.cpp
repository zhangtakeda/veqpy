#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "calculus.h"
#include "linalg.h"
#include "quadrature.h"
#include "tensor.h"

namespace
{
    using tensor::Matrix;

    int cminpack_residual(void*, int n, const double* x, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != 1) return 0;
        fvec[0] = x[0] * x[0] - 4.0;
        return 0;
    }

    double square_probe(double x) { return x * x; }

    constexpr double compile_time_sqrt_9 = gcem::sqrt(9.0);
    static_assert(compile_time_sqrt_9 > 2.999999999 && compile_time_sqrt_9 < 3.000000001);
    static_assert(decltype(quadrature::legendre::nodes)::count == quadrature::legendre::Nr);

    constexpr double legendre_weight_sum()
    {
        double total = 0.0;
        for (std::size_t i = 0; i < quadrature::legendre::Nr; ++i)
            total += quadrature::legendre::weights[i];
        return total;
    }

    constexpr double spectral_accumulate_constant(std::size_t row)
    {
        double total = 0.0;
        for (std::size_t col = 0; col < quadrature::legendre::Nr; ++col)
            total += calculus::legendre::accumulator(row, col);
        return total;
    }

    static_assert(std::abs(legendre_weight_sum() - 1.0) < 1.0e-14);
    static_assert(std::abs(spectral_accumulate_constant(31) - quadrature::legendre::nodes[31]) < 1.0e-9);

#if ENABLE_ENZYME
    extern "C" double __enzyme_autodiff(void*, double);
#endif

} // namespace

int main()
{
    nlohmann::json report;
    report["cxx"]  = static_cast<long>(__cplusplus);
    report["gcem"] = {
        {"sqrt_9", compile_time_sqrt_9},
    };
    report["quadrature"] = {
        {"Nr", quadrature::legendre::Nr},
        {"first_node", quadrature::legendre::nodes[0]},
        {"last_node", quadrature::legendre::nodes[quadrature::legendre::Nr - 1]},
        {"weight_sum", legendre_weight_sum()},
        {"nodes_aligned", quadrature::legendre::nodes.is_aligned()},
        {"weights_aligned", quadrature::legendre::weights.is_aligned()},
    };
    report["calculus"] = {
        {"accumulator_aligned", calculus::legendre::accumulator.is_aligned()},
        {"differentiator_aligned", calculus::legendre::differentiator.is_aligned()},
        {"prefix_one_at_last_node", spectral_accumulate_constant(quadrature::legendre::Nr - 1)},
    };

    Matrix<double, 32, 32> tensor_probe;
    tensor_probe(0, 0)             = 1.0;
    tensor_probe(31, 31)           = 2.0;
    const bool tensor_base_aligned = tensor_probe.is_aligned();
    report["tensor"]               = {
        {"alignment", Matrix<double, 32, 32>::alignment},
        {"base_aligned", tensor_base_aligned},
        {"storage_bytes", Matrix<double, 32, 32>::storage_bytes},
    };

    const Matrix<double, 2, 2> dense_matrix{3.0, 1.0, 1.0, 2.0};
    const Matrix<double, 2, 1> dense_rhs{9.0, 8.0};
    const auto                 doolittle_x = linalg::solve<linalg::Doolittle>(dense_matrix, dense_rhs);
    const auto                 cholesky_x  = linalg::solve<linalg::Cholesky>(dense_matrix, dense_rhs);
    const auto                 bunch_x     = linalg::solve<linalg::BunchKaufman>(dense_matrix, dense_rhs);
    const auto                 qr_x        = linalg::solve<linalg::Householder>(dense_matrix, dense_rhs);
    const auto                 svd_x       = linalg::solve<linalg::GolubReinsch>(dense_matrix, dense_rhs);
    report["linalg"]                       = {
        {"doolittle", {doolittle_x(0, 0), doolittle_x(1, 0)}},
        {"cholesky", {cholesky_x(0, 0), cholesky_x(1, 0)}},
        {"bunch_kaufman", {bunch_x(0, 0), bunch_x(1, 0)}},
        {"householder", {qr_x(0, 0), qr_x(1, 0)}},
        {"golub_reinsch", {svd_x(0, 0), svd_x(1, 0)}},
    };

    double        root_x[1]           = {3.0};
    double        root_f[1]           = {0.0};
    constexpr int root_n              = 1;
    constexpr int root_lwa            = root_n * (3 * root_n + 13) / 2;
    double        root_work[root_lwa] = {};
    const int     root_info = hybrd1(cminpack_residual, nullptr, root_n, root_x, root_f, 1.0e-10, root_work, root_lwa);

    report["cminpack"] = {
        {"info", root_info},
        {"x", root_x[0]},
        {"f", root_f[0]},
    };

    double a[4] = {
        3.0,
        1.0,
        1.0,
        2.0,
    };
    double           b[2]        = {9.0, 8.0};
    lapack_int       ipiv[2]     = {};
    const lapack_int lapack_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, a, 2, ipiv, b, 1);
    report["lapacke"]            = {
        {"info", static_cast<int>(lapack_info)},
        {"solution", {b[0], b[1]}},
    };

#if ENABLE_ENZYME
    const double derivative = __enzyme_autodiff(reinterpret_cast<void*>(square_probe), 3.0);
    report["enzyme"]        = {
        {"square_derivative_at_3", derivative},
    };
#else
    report["enzyme"] = nullptr;
#endif

    std::cout << report.dump(2) << '\n';

    const bool ok = root_info > 0 && std::abs(compile_time_sqrt_9 - 3.0) < 1.0e-12 &&
                    std::abs(legendre_weight_sum() - 1.0) < 1.0e-14 && quadrature::legendre::nodes.is_aligned() &&
                    quadrature::legendre::weights.is_aligned() && calculus::legendre::accumulator.is_aligned() &&
                    calculus::legendre::differentiator.is_aligned() && tensor_base_aligned && tensor_probe[0] == 1.0 &&
                    tensor_probe[Matrix<double, 32, 32>::count - 1] == 2.0 &&
                    std::abs(doolittle_x(0, 0) - 2.0) < 1.0e-10 && std::abs(doolittle_x(1, 0) - 3.0) < 1.0e-10 &&
                    std::abs(cholesky_x(0, 0) - 2.0) < 1.0e-10 && std::abs(cholesky_x(1, 0) - 3.0) < 1.0e-10 &&
                    std::abs(bunch_x(0, 0) - 2.0) < 1.0e-10 && std::abs(bunch_x(1, 0) - 3.0) < 1.0e-10 &&
                    std::abs(qr_x(0, 0) - 2.0) < 1.0e-10 && std::abs(qr_x(1, 0) - 3.0) < 1.0e-10 &&
                    std::abs(svd_x(0, 0) - 2.0) < 1.0e-10 && std::abs(svd_x(1, 0) - 3.0) < 1.0e-10 &&
                    std::abs(root_x[0] - 2.0) < 1.0e-8 && lapack_info == 0 && std::abs(b[0] - 2.0) < 1.0e-10 &&
                    std::abs(b[1] - 3.0) < 1.0e-10
#if ENABLE_ENZYME
                    && std::abs(report["enzyme"]["square_derivative_at_3"].get<double>() - 6.0) < 1.0e-8
#endif
        ;

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
