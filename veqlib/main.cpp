#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "grid.h"
#include "linalg.h"
#include "math.h"
#include "test_cli.h"
#include "tensor.h"

namespace
{
    using tensor::Matrix;

    int cminpack_residual(void*, int n, const double* x, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != 1)
            return 0;
        fvec[0] = x[0] * x[0] - 4.0;
        return 0;
    }

    double square_probe(double x) { return x * x; }

    constexpr double               compile_time_sqrt_9        = gcem::sqrt(9.0);
    constexpr auto                 chebyshev_nodes_8          = grid::Chebyshev::nodes<8>;
    constexpr auto                 chebyshev_weights_8        = grid::Chebyshev::weights<8>;
    constexpr auto                 legendre_nodes_32          = grid::Legendre::nodes<32>;
    constexpr auto                 legendre_weights_32        = grid::Legendre::weights<32>;
    constexpr auto                 lobatto_nodes_8            = grid::Lobatto::nodes<8>;
    constexpr auto                 lobatto_weights_8          = grid::Lobatto::weights<8>;
    constexpr auto                 radau_nodes_8              = grid::Radau::nodes<8>;
    constexpr auto                 radau_weights_8            = grid::Radau::weights<8>;
    constexpr auto                 radau_weights_256          = grid::Radau::weights<256>;
    constexpr auto                 legendre_accumulator_32    = grid::Spectral::accumulator<32, grid::Legendre>;
    constexpr auto                 legendre_differentiator_32 = grid::Spectral::differentiator<32, grid::Legendre>;
    constexpr auto                 cfd33_accumulator_8        = grid::CFD33::accumulator<8, grid::Lobatto>;
    constexpr auto                 cfd33_differentiator_8     = grid::CFD33::differentiator<8, grid::Lobatto>;
    constexpr auto                 cfd35_accumulator_8        = grid::CFD35::accumulator<8, grid::Lobatto>;
    constexpr auto                 cfd35_differentiator_8     = grid::CFD35::differentiator<8, grid::Lobatto>;
    constexpr auto                 cfd55_accumulator_8        = grid::CFD55::accumulator<8, grid::Lobatto>;
    constexpr auto                 cfd55_differentiator_8     = grid::CFD55::differentiator<8, grid::Lobatto>;
    constexpr Matrix<double, 3, 3> thomas_band{0.0, -1.0, -1.0, 2.0, 2.0, 2.0, -1.0, -1.0, 0.0};
    constexpr Matrix<double, 3, 1> thomas_rhs{1.0, 0.0, 1.0};
    constexpr auto                 thomas_solution = linalg::solve<linalg::Thomas>(thomas_band, thomas_rhs);
    static_assert(compile_time_sqrt_9 > 2.999999999 && compile_time_sqrt_9 < 3.000000001);
    static_assert(chebyshev_nodes_8[0] > 0.0 && chebyshev_nodes_8[7] < 1.0);
    static_assert(lobatto_nodes_8[0] == 0.0 && lobatto_nodes_8[7] == 1.0);
    static_assert(radau_nodes_8[0] > 0.0 && radau_nodes_8[7] == 1.0);

    constexpr double legendre_weight_sum()
    {
        double total = 0.0;
        for (std::size_t i = 0; i < decltype(legendre_weights_32)::count; ++i)
            total += legendre_weights_32[i];
        return total;
    }

    template <typename Values>
    constexpr double sum_values(const Values& values)
    {
        double total = 0.0;
        for (std::size_t i = 0; i < Values::count; ++i)
            total += values[i];
        return total;
    }

    constexpr double spectral_accumulate_constant(std::size_t row)
    {
        const auto* values = legendre_accumulator_32.data();
        double      total  = 0.0;
        for (std::size_t col = 0; col < decltype(legendre_nodes_32)::count; ++col)
            total += values[row * decltype(legendre_nodes_32)::count + col];
        return total;
    }

    constexpr double spectral_differentiate_identity(std::size_t row)
    {
        const auto* values = legendre_differentiator_32.data();
        double      total  = 0.0;
        for (std::size_t col = 0; col < decltype(legendre_nodes_32)::count; ++col)
            total += values[row * decltype(legendre_nodes_32)::count + col] * legendre_nodes_32[col];
        return total;
    }

    constexpr double thomas_factorize_substitute_sum()
    {
        auto       rhs     = thomas_rhs;
        const auto context = linalg::factorize<linalg::Thomas>(thomas_band);
        context.substitute_inplace<1>(rhs.data());
        return rhs[0] + rhs[1] + rhs[2];
    }

    template <typename Operator, typename Nodes>
    constexpr double apply_to_identity(const Operator& op, const Nodes& nodes, std::size_t row)
    {
        const auto* values = op.data();
        double      total  = 0.0;
        for (std::size_t col = 0; col < Nodes::count; ++col)
            total += values[row * Nodes::count + col] * nodes[col];
        return total;
    }

    template <typename Operator>
    constexpr double apply_to_constant(const Operator& op, std::size_t row)
    {
        const auto* values = op.data();
        double      total  = 0.0;
        for (std::size_t col = 0; col < Operator::shape[1]; ++col)
            total += values[row * Operator::shape[1] + col];
        return total;
    }

    static_assert(std::abs(legendre_weight_sum() - 1.0) < 1.0e-14);
    static_assert(std::abs(sum_values(chebyshev_weights_8) - 1.0) < 1.0e-14);
    static_assert(std::abs(sum_values(legendre_weights_32) - 1.0) < 1.0e-14);
    static_assert(std::abs(sum_values(lobatto_weights_8) - 1.0) < 1.0e-14);
    static_assert(std::abs(sum_values(radau_weights_8) - 1.0) < 1.0e-14);
    static_assert(std::abs(sum_values(radau_weights_256) - 1.0) < 1.0e-14);
    static_assert(std::abs(spectral_accumulate_constant(31) - legendre_nodes_32[31]) < 1.0e-9);
    static_assert(std::abs(spectral_differentiate_identity(0) - 1.0) < 1.0e-9);
    static_assert(std::abs(spectral_differentiate_identity(31) - 1.0) < 1.0e-9);
    static_assert(std::abs(apply_to_identity(cfd33_differentiator_8, lobatto_nodes_8, 0) - 1.0) < 1.0e-8);
    static_assert(std::abs(apply_to_identity(cfd35_differentiator_8, lobatto_nodes_8, 0) - 1.0) < 1.0e-8);
    static_assert(std::abs(apply_to_identity(cfd55_differentiator_8, lobatto_nodes_8, 0) - 1.0) < 1.0e-8);
    static_assert(std::abs(apply_to_constant(cfd33_accumulator_8, 7) - lobatto_nodes_8[7]) < 1.0e-8);
    static_assert(std::abs(apply_to_constant(cfd35_accumulator_8, 7) - lobatto_nodes_8[7]) < 1.0e-8);
    static_assert(std::abs(apply_to_constant(cfd55_accumulator_8, 7) - lobatto_nodes_8[7]) < 1.0e-8);
    static_assert(std::abs(thomas_solution[0] - 1.0) < 1.0e-12);
    static_assert(std::abs(thomas_solution[1] - 1.0) < 1.0e-12);
    static_assert(std::abs(thomas_solution[2] - 1.0) < 1.0e-12);
    static_assert(std::abs(thomas_factorize_substitute_sum() - 3.0) < 1.0e-12);

#if ENABLE_ENZYME
    extern "C" double __enzyme_autodiff(void*, double);
#endif

} // namespace

int run_probe(int, char**)
{
    nlohmann::json report;
    report["cxx"]  = static_cast<long>(__cplusplus);
    report["gcem"] = {
        {"sqrt_9", compile_time_sqrt_9},
    };
    report["grid"] = {
        {"Nr", decltype(legendre_nodes_32)::count},
        {"first_node", legendre_nodes_32[0]},
        {"last_node", legendre_nodes_32[decltype(legendre_nodes_32)::count - 1]},
        {"generated_radau_256_weight_sum", sum_values(radau_weights_256)},
        {"weight_sum", legendre_weight_sum()},
    };
    report["spectral"] = {
        {"identity_derivative_first_node", spectral_differentiate_identity(0)},
        {"identity_derivative_last_node", spectral_differentiate_identity(decltype(legendre_nodes_32)::count - 1)},
        {"prefix_one_at_last_node", spectral_accumulate_constant(decltype(legendre_nodes_32)::count - 1)},
    };
    report["cfd"] = {
        {"cfd33_identity_derivative", apply_to_identity(cfd33_differentiator_8, lobatto_nodes_8, 0)},
        {"cfd35_identity_derivative", apply_to_identity(cfd35_differentiator_8, lobatto_nodes_8, 0)},
        {"cfd55_identity_derivative", apply_to_identity(cfd55_differentiator_8, lobatto_nodes_8, 0)},
        {"cfd33_prefix_one_at_last_node", apply_to_constant(cfd33_accumulator_8, 7)},
        {"cfd35_prefix_one_at_last_node", apply_to_constant(cfd35_accumulator_8, 7)},
        {"cfd55_prefix_one_at_last_node", apply_to_constant(cfd55_accumulator_8, 7)},
    };

    Matrix<double, 32, 32> tensor_probe;
    tensor_probe(0, 0)   = 1.0;
    tensor_probe(31, 31) = 2.0;
    report["tensor"]     = {
        {"alignment", Matrix<double, 32, 32>::alignment},
        {"finite", math::is_finite(tensor_probe)},
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
        {"thomas", {thomas_solution(0, 0), thomas_solution(1, 0), thomas_solution(2, 0)}},
    };

    double        root_x[1] = {3.0};
    double        root_f[1] = {0.0};
    constexpr int root_n    = 1;
    constexpr int root_lwa  = root_n * (3 * root_n + 13) / 2;
    double        root_work[root_lwa];
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
    double           b[2] = {9.0, 8.0};
    lapack_int       ipiv[2];
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
                    std::abs(legendre_weight_sum() - 1.0) < 1.0e-14 && math::is_finite(tensor_probe) &&
                    tensor_probe[0] == 1.0 && tensor_probe[Matrix<double, 32, 32>::count - 1] == 2.0 &&
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

namespace
{
    using CliEntrypoint = int (*)(int, char**);

    void print_usage()
    {
        std::cout
            << "usage: veqlib_main [--mode MODE|MODE] [mode options]\n"
               "\n"
               "modes:\n"
               "  probe              dependency and constexpr smoke test (default)\n"
               "  temp-validation    generated-topology validation suite\n"
               "  pf-validation      PF/psin/uniform/Ip C++ vs VEQPy validation payload\n"
               "  solve              PF/psin/uniform/Ip solve benchmark\n"
               "  stage              PF/psin/uniform/Ip stage benchmark\n"
               "\n"
               "examples:\n"
               "  veqlib_main --mode solve --repeat 30 --warmup 5\n"
               "  veqlib_main --mode solve --scan-points 9 --scan-policy all\n"
               "  veqlib_main --mode stage --stage evaluate --repeat 10 --inner 10000\n";
    }

    int run_forwarded(CliEntrypoint entrypoint, int argc, char** argv, int first_arg)
    {
        std::vector<char*> forwarded;
        forwarded.reserve(static_cast<std::size_t>(argc - first_arg + 2));
        forwarded.push_back(argv[0]);
        for (int i = first_arg; i < argc; ++i)
            forwarded.push_back(argv[i]);
        forwarded.push_back(nullptr);
        return entrypoint(static_cast<int>(forwarded.size() - 1), forwarded.data());
    }

    int run_mode(const std::string& mode, int argc, char** argv, int first_arg)
    {
        if (mode == "probe")
            return run_forwarded(run_probe, argc, argv, first_arg);
        if (mode == "temp-validation" || mode == "temp" || mode == "topology-validation")
            return run_forwarded(veqlib_temp_validation_cli::run, argc, argv, first_arg);
        if (mode == "pf-validation" || mode == "validate" || mode == "validation")
            return run_forwarded(veqlib_pf_psin_uniform_validation_cli::run, argc, argv, first_arg);
        if (mode == "solve" || mode == "solve-benchmark" || mode == "pf-benchmark" || mode == "benchmark")
            return run_forwarded(veqlib_pf_psin_uniform_benchmark_cli::run, argc, argv, first_arg);
        if (mode == "stage" || mode == "stage-benchmark")
            return run_forwarded(veqlib_stage_benchmark_cli::run, argc, argv, first_arg);

        std::cerr << "veqlib_main: unknown mode: " << mode << '\n';
        print_usage();
        return 2;
    }

    bool is_solve_option(const std::string& option)
    {
        return option == "--repeat" || option == "--warmup" || option == "--solver" || option == "--enzyme-width" ||
               option == "--jacobian-check" || option == "--scan-points" || option == "--scan-policy" ||
               option == "--scan-relative-step" || option == "--scan-step";
    }

    bool is_stage_option(const std::string& option)
    {
        return option == "--stage" || option == "--backend" || option == "--inner" || option == "--ring-size";
    }
} // namespace

int main(int argc, char** argv)
{
    if (argc <= 1)
        return run_mode("probe", argc, argv, 1);

    const std::string first = argv[1];
    if (first == "--help" || first == "-h")
    {
        print_usage();
        return 0;
    }
    if (first == "--mode")
    {
        if (argc <= 2)
        {
            std::cerr << "veqlib_main: --mode requires a value\n";
            return 2;
        }
        return run_mode(argv[2], argc, argv, 3);
    }
    if (first.rfind("--mode=", 0) == 0)
        return run_mode(first.substr(7), argc, argv, 2);
    if (is_stage_option(first))
        return run_mode("stage", argc, argv, 1);
    if (is_solve_option(first))
        return run_mode("solve", argc, argv, 1);

    return run_mode(first, argc, argv, 2);
}
