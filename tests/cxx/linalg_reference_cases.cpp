// Differential cases for the fixed-size VEQ linalg policies.
//
// The driver compares the in-tree fixed-size templates with row-major LAPACKE
// reference entry points.  The matrices are deterministic and well
// conditioned, so a mismatch identifies an algorithmic-semantic regression
// rather than a conditioning accident.

#include "linalg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include <cblas.h>
#include <lapacke.h>

namespace
{
    template <std::size_t Rows, std::size_t Cols>
    using Mat = tensor::Matrix<double, Rows, Cols>;

    constexpr auto constexpr_golub_reinsch_reference()
    {
        Mat<2, 2> matrix{tensor::uninitialized};
        matrix(0, 0) = 3.0;
        matrix(0, 1) = 0.0;
        matrix(1, 0) = 0.0;
        matrix(1, 1) = 2.0;
        return linalg::factorize<linalg::GolubReinsch>(matrix);
    }

    constexpr auto constexpr_svd = constexpr_golub_reinsch_reference();
    static_assert(constexpr_svd.info == 0);
    static_assert(constexpr_svd.S_vec[0] == 3.0 && constexpr_svd.S_vec[1] == 2.0);

#if defined(VEQPY_CXX_FORCE_INTERNAL_LINALG)
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::Doolittle, 128, 128>);
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::GolubReinsch, 128, 64>);
#else
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::Doolittle, 127, 127>);
    static_assert(linalg::detail::uses_runtime_lapack<linalg::Doolittle, 128, 128>);
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::BunchKaufman, 79, 79>);
    static_assert(linalg::detail::uses_runtime_lapack<linalg::BunchKaufman, 80, 80>);
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::Cholesky, 128, 128>);
    static_assert(linalg::detail::uses_runtime_lapack<linalg::Cholesky, 129, 129>);
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::Householder, 128, 64>);
    static_assert(linalg::detail::uses_runtime_lapack<linalg::Householder, 129, 64>);
    static_assert(!linalg::detail::uses_runtime_lapack<linalg::GolubReinsch, 63, 32>);
    static_assert(linalg::detail::uses_runtime_lapack<linalg::GolubReinsch, 64, 32>);
#endif

    template <std::size_t Rows, std::size_t Cols>
    double relative_error(const Mat<Rows, Cols>& got, const Mat<Rows, Cols>& expected)
    {
        double maximum_error = 0.0;
        double maximum_value = 0.0;
        for (std::size_t index = 0; index < Rows * Cols; ++index)
        {
            maximum_error = std::max(maximum_error, std::abs(got[index] - expected[index]));
            maximum_value = std::max(maximum_value, std::abs(expected[index]));
        }
        return maximum_error / std::max(maximum_value, std::numeric_limits<double>::min());
    }

    template <std::size_t N>
    double lower_relative_error(const double* got, const double* expected)
    {
        double maximum_error = 0.0;
        double maximum_value = 0.0;
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column <= row; ++column)
            {
                maximum_error = std::max(maximum_error, std::abs(got[row * N + column] - expected[row * N + column]));
                maximum_value = std::max(maximum_value, std::abs(expected[row * N + column]));
            }
        return maximum_error / std::max(maximum_value, std::numeric_limits<double>::min());
    }

    template <std::size_t Count>
    double relative_error(const double* got, const double* expected)
    {
        double maximum_error = 0.0;
        double maximum_value = 0.0;
        for (std::size_t index = 0; index < Count; ++index)
        {
            maximum_error = std::max(maximum_error, std::abs(got[index] - expected[index]));
            maximum_value = std::max(maximum_value, std::abs(expected[index]));
        }
        return maximum_error / std::max(maximum_value, std::numeric_limits<double>::min());
    }

    constexpr double entry(std::size_t row, std::size_t column, std::size_t salt)
    {
        const auto value = (row * 37 + column * 17 + salt * 13 + 11) % 41;
        return (static_cast<double>(value) - 20.0) / 41.0;
    }

    template <std::size_t Rows, std::size_t Columns, std::size_t P>
    void form_rhs(const Mat<Rows, Columns>& matrix, const Mat<Columns, P>& expected, Mat<Rows, P>& rhs)
    {
        for (std::size_t row = 0; row < Rows; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
            {
                double value = 0.0;
                for (std::size_t column = 0; column < Columns; ++column)
                    value += matrix(row, column) * expected(column, rhs_column);
                rhs(row, rhs_column) = value;
            }
    }

    template <std::size_t N, std::size_t P>
    void make_general_system(Mat<N, N>& matrix, Mat<N, P>& rhs, Mat<N, P>& expected)
    {
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column < N; ++column)
                matrix(row, column) = entry(row, column, 1);
        for (std::size_t diagonal = 0; diagonal < N; ++diagonal)
            matrix(diagonal, diagonal) += static_cast<double>(N);
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                expected(row, rhs_column) = 0.125 + 0.0625 * static_cast<double>((row + 3 * rhs_column) % 11);
        form_rhs(matrix, expected, rhs);
    }

    template <std::size_t N, std::size_t P>
    void make_spd_system(Mat<N, N>& matrix, Mat<N, P>& rhs, Mat<N, P>& expected)
    {
        Mat<N, N> basis{tensor::uninitialized};
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column < N; ++column)
                basis(row, column) = entry(row, column, 2);
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column < N; ++column)
            {
                double value = 0.0;
                for (std::size_t k = 0; k < N; ++k)
                    value += basis(row, k) * basis(column, k);
                matrix(row, column) = value / static_cast<double>(N) + (row == column ? 1.0 : 0.0);
            }
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                expected(row, rhs_column) = 0.1 + 0.075 * static_cast<double>((2 * row + rhs_column) % 7);
        form_rhs(matrix, expected, rhs);
    }

    template <std::size_t N, std::size_t P>
    void make_symmetric_indefinite_system(Mat<N, N>& matrix, Mat<N, P>& rhs, Mat<N, P>& expected)
    {
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column <= row; ++column)
            {
                const double value = entry(row, column, 3);
                matrix(row, column) = value;
                matrix(column, row) = value;
            }
        for (std::size_t diagonal = 0; diagonal < N; ++diagonal)
            matrix(diagonal, diagonal) += (diagonal % 2 == 0 ? 1.0 : -1.0) * static_cast<double>(N);
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                expected(row, rhs_column) = 0.1 + 0.05 * static_cast<double>((row + 5 * rhs_column) % 13);
        form_rhs(matrix, expected, rhs);
    }

    template <std::size_t M, std::size_t N, std::size_t P>
    void make_qr_system(Mat<M, N>& matrix, Mat<M, P>& rhs, Mat<N, P>& expected)
    {
        for (std::size_t row = 0; row < M; ++row)
            for (std::size_t column = 0; column < N; ++column)
                matrix(row, column) = entry(row, column, 4);
        for (std::size_t diagonal = 0; diagonal < N; ++diagonal)
            matrix(diagonal, diagonal) += static_cast<double>(N);
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                expected(row, rhs_column) = 0.1 + 0.0625 * static_cast<double>((3 * row + rhs_column) % 9);
        for (std::size_t row = 0; row < M; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
            {
                double value = 0.0;
                for (std::size_t column = 0; column < N; ++column)
                    value += matrix(row, column) * expected(column, rhs_column);
                rhs(row, rhs_column) = value;
            }
    }

    void emit(const char* name, double internal_error, double lapack_error, double delta)
    {
        std::cout << std::setprecision(17) << name << ' ' << internal_error << ' ' << lapack_error << ' ' << delta
                  << '\n';
    }

    template <std::size_t N, std::size_t P>
    void run_doolittle()
    {
        Mat<N, N> matrix{tensor::uninitialized};
        Mat<N, P> rhs{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> internal{tensor::uninitialized};
        Mat<N, P> lapack{tensor::uninitialized};
        make_general_system(matrix, rhs, expected);

        linalg::Context<linalg::Doolittle, N, N> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, internal.data());
        context.template substitute_inplace<P>(internal.data());

        std::array<double, N * N> factor{};
        std::array<lapack_int, N> pivots{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack.data());
        if (LAPACKE_dgetrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(N), static_cast<lapack_int>(N), factor.data(),
                           static_cast<lapack_int>(N), pivots.data()) != 0 ||
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', static_cast<lapack_int>(N), static_cast<lapack_int>(P), factor.data(),
                           static_cast<lapack_int>(N), pivots.data(), lapack.data(), static_cast<lapack_int>(P)) != 0)
            std::abort();

        emit("doolittle", relative_error(internal, expected), relative_error(lapack, expected),
             relative_error(internal, lapack));
    }

    template <std::size_t N, std::size_t P>
    void run_cholesky()
    {
        Mat<N, N> matrix{tensor::uninitialized};
        Mat<N, P> rhs{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> internal{tensor::uninitialized};
        Mat<N, P> lapack{tensor::uninitialized};
        make_spd_system(matrix, rhs, expected);

        linalg::Context<linalg::Cholesky, N, N> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, internal.data());
        context.template substitute_inplace<P>(internal.data());

        std::array<double, N * N> factor{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack.data());
        if (LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(),
                           static_cast<lapack_int>(N)) != 0 ||
            LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), static_cast<lapack_int>(P), factor.data(),
                           static_cast<lapack_int>(N), lapack.data(), static_cast<lapack_int>(P)) != 0)
            std::abort();

        emit("cholesky", relative_error(internal, expected), relative_error(lapack, expected),
             relative_error(internal, lapack));
    }

    template <std::size_t N, std::size_t P>
    void run_bunch_kaufman()
    {
        Mat<N, N> matrix{tensor::uninitialized};
        Mat<N, P> rhs{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> internal{tensor::uninitialized};
        Mat<N, P> lapack{tensor::uninitialized};
        make_symmetric_indefinite_system(matrix, rhs, expected);

        linalg::Context<linalg::BunchKaufman, N, N> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, internal.data());
        context.template substitute_inplace<P>(internal.data());

        std::array<double, N * N> factor{};
        std::array<lapack_int, N> pivots{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack.data());
        if (LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(),
                           static_cast<lapack_int>(N), pivots.data()) != 0 ||
            LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), static_cast<lapack_int>(P), factor.data(),
                           static_cast<lapack_int>(N), pivots.data(), lapack.data(), static_cast<lapack_int>(P)) != 0)
            std::abort();

        emit("bunch_kaufman", relative_error(internal, expected), relative_error(lapack, expected),
             relative_error(internal, lapack));
    }

    template <std::size_t M, std::size_t N, std::size_t P>
    void run_householder()
    {
        Mat<M, N> matrix{tensor::uninitialized};
        Mat<M, P> rhs{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> internal{tensor::uninitialized};
        Mat<N, P> lapack{tensor::uninitialized};
        Mat<M, P> internal_work{tensor::uninitialized};
        Mat<M, P> lapack_work{tensor::uninitialized};
        make_qr_system(matrix, rhs, expected);

        linalg::Context<linalg::Householder, M, N> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, internal_work.data());
        context.template substitute_inplace<P>(internal_work.data());
        std::copy(internal_work.data(), internal_work.data() + internal.count, internal.data());

        std::array<double, M * N> factor{};
        std::array<double, N>     tau{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack_work.data());
        if (LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(M), static_cast<lapack_int>(N), factor.data(),
                           static_cast<lapack_int>(N), tau.data()) != 0 ||
            LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', static_cast<lapack_int>(M), static_cast<lapack_int>(P),
                           static_cast<lapack_int>(N), factor.data(), static_cast<lapack_int>(N), tau.data(),
                           lapack_work.data(), static_cast<lapack_int>(P)) != 0)
            std::abort();
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, static_cast<int>(N),
                    static_cast<int>(P), 1.0, factor.data(), static_cast<int>(N), lapack_work.data(), static_cast<int>(P));
        std::copy(lapack_work.data(), lapack_work.data() + lapack.count, lapack.data());

        emit("householder", relative_error(internal, expected), relative_error(lapack, expected),
             relative_error(internal, lapack));
    }

    void run_lu_subnormal_pivot()
    {
        constexpr std::size_t N = 2;
        Mat<N, N> matrix{tensor::uninitialized};
        matrix(0, 0) = 1.0e-310;
        matrix(0, 1) = 1.0;
        matrix(1, 0) = 1.0e-311;
        matrix(1, 1) = 1.0;

        linalg::Context<linalg::Doolittle, N, N> context;
        context.factorize_from(matrix.data());

        std::array<double, N * N> factor{};
        std::array<lapack_int, N> pivots{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dgetf2(
            LAPACK_ROW_MAJOR, static_cast<lapack_int>(N), static_cast<lapack_int>(N), factor.data(),
            static_cast<lapack_int>(N), pivots.data());
        if (info < 0)
            std::abort();

        double pivot_error = 0.0;
        for (std::size_t index = 0; index < N; ++index)
            pivot_error = std::max(pivot_error, std::abs(static_cast<double>(context.pivot_vec[index] + 1 - pivots[index])));
        // The bundled LAPACKE build reports the same pivot/info but flushes
        // its subnormal DGETF2 update on this host.  Check the source-kernel
        // branch directly: DGETF2 divides here rather than forming an
        // overflowing reciprocal.
        emit("doolittle_subnormal", std::abs(static_cast<double>(context.info - info)),
             std::abs(context.LU_mat(1, 0) - 0.1), pivot_error);
    }

    void run_cholesky_lower_storage()
    {
        constexpr std::size_t N = 3;
        Mat<N, N> matrix{tensor::uninitialized};
        matrix(0, 0) = 4.0;
        matrix(0, 1) = -7.0;
        matrix(0, 2) = 19.0;
        matrix(1, 0) = 1.0;
        matrix(1, 1) = 3.0;
        matrix(1, 2) = -13.0;
        matrix(2, 0) = -2.0;
        matrix(2, 1) = 0.5;
        matrix(2, 2) = 5.0;

        linalg::Context<linalg::Cholesky, N, N> context;
        context.factorize_from(matrix.data());

        std::array<double, N * N> factor{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dpotrf(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(), static_cast<lapack_int>(N));
        if (info < 0)
            std::abort();

        emit("cholesky_lower_storage", std::abs(static_cast<double>(context.info - info)),
             lower_relative_error<N>(context.LLT_mat.data(), factor.data()), 0.0);
    }

    void run_cholesky_non_positive()
    {
        constexpr std::size_t N = 2;
        Mat<N, N> matrix{tensor::uninitialized};
        matrix(0, 0) = 1.0;
        matrix(0, 1) = 5.0;
        matrix(1, 0) = 0.0;
        matrix(1, 1) = -1.0;

        linalg::Context<linalg::Cholesky, N, N> context;
        context.factorize_from(matrix.data());

        std::array<double, N * N> factor{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dpotrf(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(), static_cast<lapack_int>(N));
        if (info < 0)
            std::abort();

        emit("cholesky_non_positive", std::abs(static_cast<double>(context.info - info)),
             lower_relative_error<N>(context.LLT_mat.data(), factor.data()), 0.0);
    }

    void run_bunch_kaufman_two_by_two()
    {
        constexpr std::size_t N = 4;
        constexpr std::size_t P = 1;
        Mat<N, N> matrix{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> rhs{tensor::uninitialized};
        matrix(0, 0) = 0.0;
        matrix(0, 1) = 1.0;
        matrix(0, 2) = 0.0;
        matrix(0, 3) = 0.0;
        matrix(1, 0) = 1.0;
        matrix(1, 1) = 0.0;
        matrix(1, 2) = 0.125;
        matrix(1, 3) = 0.0;
        matrix(2, 0) = 0.0;
        matrix(2, 1) = 0.125;
        matrix(2, 2) = 2.0;
        matrix(2, 3) = 0.25;
        matrix(3, 0) = 0.0;
        matrix(3, 1) = 0.0;
        matrix(3, 2) = 0.25;
        matrix(3, 3) = 3.0;
        for (std::size_t row = 0; row < N; ++row)
            expected(row, 0) = 0.25 + 0.125 * static_cast<double>(row);
        form_rhs(matrix, expected, rhs);

        linalg::Context<linalg::BunchKaufman, N, N> context;
        context.factorize_from(matrix.data());
        Mat<N, P> internal{tensor::uninitialized};
        std::copy(rhs.data(), rhs.data() + rhs.count, internal.data());
        context.template substitute_inplace<P>(internal.data());

        std::array<double, N * N> factor{};
        std::array<lapack_int, N> pivots{};
        Mat<N, P> lapack{tensor::uninitialized};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack.data());
        const lapack_int factor_info = LAPACKE_dsytrf(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(), static_cast<lapack_int>(N), pivots.data());
        const lapack_int solve_info = LAPACKE_dsytrs(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), static_cast<lapack_int>(P), factor.data(),
            static_cast<lapack_int>(N), pivots.data(), lapack.data(), static_cast<lapack_int>(P));
        if (factor_info != 0 || solve_info != 0)
            std::abort();

        double pivot_error = 0.0;
        for (std::size_t index = 0; index < N; ++index)
            pivot_error = std::max(pivot_error, std::abs(static_cast<double>(context.pivot_vec[index] - pivots[index])));
        emit("bunch_kaufman_two_by_two", relative_error(internal, expected),
             lower_relative_error<N>(context.LDLT_mat.data(), factor.data()), pivot_error);
    }

    void run_bunch_kaufman_one_by_one_swap()
    {
        constexpr std::size_t N = 3;
        constexpr std::size_t P = 1;
        Mat<N, N> matrix{tensor::uninitialized};
        Mat<N, P> expected{tensor::uninitialized};
        Mat<N, P> rhs{tensor::uninitialized};
        matrix(0, 0) = 0.1;
        matrix(0, 1) = 0.5;
        matrix(0, 2) = 2.0;
        matrix(1, 0) = 0.5;
        matrix(1, 1) = 1.0;
        matrix(1, 2) = 0.125;
        matrix(2, 0) = 2.0;
        matrix(2, 1) = 0.125;
        matrix(2, 2) = 5.0;
        for (std::size_t row = 0; row < N; ++row)
            expected(row, 0) = 0.2 + 0.1 * static_cast<double>(row);
        form_rhs(matrix, expected, rhs);

        linalg::Context<linalg::BunchKaufman, N, N> context;
        context.factorize_from(matrix.data());
        Mat<N, P> internal{tensor::uninitialized};
        std::copy(rhs.data(), rhs.data() + rhs.count, internal.data());
        context.template substitute_inplace<P>(internal.data());

        std::array<double, N * N> factor{};
        std::array<lapack_int, N> pivots{};
        Mat<N, P> lapack{tensor::uninitialized};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, lapack.data());
        const lapack_int factor_info = LAPACKE_dsytrf(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), factor.data(), static_cast<lapack_int>(N), pivots.data());
        const lapack_int solve_info = LAPACKE_dsytrs(
            LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(N), static_cast<lapack_int>(P), factor.data(),
            static_cast<lapack_int>(N), pivots.data(), lapack.data(), static_cast<lapack_int>(P));
        if (factor_info != 0 || solve_info != 0)
            std::abort();

        double pivot_error = 0.0;
        for (std::size_t index = 0; index < N; ++index)
            pivot_error = std::max(pivot_error, std::abs(static_cast<double>(context.pivot_vec[index] - pivots[index])));
        emit("bunch_kaufman_one_by_one_swap", relative_error(internal, expected),
             lower_relative_error<N>(context.LDLT_mat.data(), factor.data()), pivot_error);
    }

    void run_householder_subnormal_reflector()
    {
        constexpr std::size_t M = 3;
        constexpr std::size_t N = 1;
        Mat<M, N> matrix{tensor::uninitialized};
        matrix(0, 0) = 1.0e-310;
        matrix(1, 0) = 2.0e-310;
        matrix(2, 0) = -1.0e-310;

        linalg::Context<linalg::Householder, M, N> context;
        context.factorize_from(matrix.data());

        std::array<double, M * N> factor{};
        std::array<double, N> tau{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dgeqrf(
            LAPACK_ROW_MAJOR, static_cast<lapack_int>(M), static_cast<lapack_int>(N), factor.data(),
            static_cast<lapack_int>(N), tau.data());
        if (info != 0)
            std::abort();

        emit("householder_subnormal", std::abs(static_cast<double>(context.info - info)),
             relative_error<M * N>(context.QR_mat.data(), factor.data()), relative_error<N>(context.tau_vec.data(), tau.data()));
    }

    template <std::size_t Dimension>
    double orthogonality_error(const double* matrix)
    {
        double maximum_error = 0.0;
        for (std::size_t left = 0; left < Dimension; ++left)
            for (std::size_t right = 0; right < Dimension; ++right)
            {
                double value = 0.0;
                for (std::size_t row = 0; row < Dimension; ++row)
                    value += matrix[row * Dimension + left] * matrix[row * Dimension + right];
                maximum_error = std::max(maximum_error, std::abs(value - (left == right ? 1.0 : 0.0)));
            }
        return maximum_error;
    }

    template <std::size_t Rows, std::size_t Columns>
    double svd_reconstruction_error(const Mat<Rows, Columns>& matrix,
                                    const linalg::Context<linalg::GolubReinsch, Rows, Columns>& context)
    {
        constexpr std::size_t rank = std::min(Rows, Columns);
        double maximum_error = 0.0;
        double maximum_value = 0.0;
        for (std::size_t row = 0; row < Rows; ++row)
            for (std::size_t column = 0; column < Columns; ++column)
            {
                double value = 0.0;
                for (std::size_t index = 0; index < rank; ++index)
                    value += context.U_mat(row, index) * context.S_vec[index] * context.Vt_mat(index, column);
                maximum_error = std::max(maximum_error, std::abs(value - matrix(row, column)));
                maximum_value = std::max(maximum_value, std::abs(matrix(row, column)));
            }
        return maximum_error / std::max(maximum_value, std::numeric_limits<double>::min());
    }

    template <std::size_t Rows, std::size_t Columns, std::size_t P>
    void run_golub_reinsch()
    {
        constexpr std::size_t rank = std::min(Rows, Columns);
        constexpr std::size_t work_rows = Rows > Columns ? Rows : Columns;
        Mat<Rows, Columns> matrix{tensor::uninitialized};
        Mat<Columns, P> expected{tensor::uninitialized};
        Mat<Rows, P> rhs{tensor::uninitialized};
        Mat<work_rows, P> work{};
        Mat<Columns, P> internal{tensor::uninitialized};
        for (std::size_t row = 0; row < Rows; ++row)
            for (std::size_t column = 0; column < Columns; ++column)
                matrix(row, column) = entry(row, column, 7) + (row == column ? 2.0 : 0.0);
        if constexpr (Rows >= Columns)
        {
            for (std::size_t row = 0; row < Columns; ++row)
                for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                    expected(row, rhs_column) = 0.125 + 0.05 * static_cast<double>((2 * row + rhs_column) % 11);
        }
        else
        {
            Mat<Rows, P> row_space_seed{tensor::uninitialized};
            for (std::size_t row = 0; row < Rows; ++row)
                for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                    row_space_seed(row, rhs_column) =
                        0.125 + 0.05 * static_cast<double>((2 * row + rhs_column) % 11);
            for (std::size_t row = 0; row < Columns; ++row)
                for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
                {
                    double value = 0.0;
                    for (std::size_t column = 0; column < Rows; ++column)
                        value += matrix(column, row) * row_space_seed(column, rhs_column);
                    expected(row, rhs_column) = value;
                }
        }
        form_rhs(matrix, expected, rhs);

        linalg::Context<linalg::GolubReinsch, Rows, Columns> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, work.data());
        context.template substitute_inplace<P>(work.data());
        std::copy(work.data(), work.data() + internal.count, internal.data());

        std::array<double, Rows * Columns> factor{};
        std::array<double, Rows * Rows> lapack_u{};
        std::array<double, Columns * Columns> lapack_vt{};
        std::array<double, rank> lapack_s{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dgesdd(
            LAPACK_ROW_MAJOR, 'A', static_cast<lapack_int>(Rows), static_cast<lapack_int>(Columns), factor.data(),
            static_cast<lapack_int>(Columns), lapack_s.data(), lapack_u.data(), static_cast<lapack_int>(Rows),
            lapack_vt.data(), static_cast<lapack_int>(Columns));
        if (info != 0)
            std::abort();

        const double orthogonality = std::max(orthogonality_error<Rows>(context.U_mat.data()),
                                              orthogonality_error<Columns>(context.Vt_mat.data()));
        const double factor_error = std::max(svd_reconstruction_error(matrix, context), orthogonality);
        emit("golub_reinsch", std::max(factor_error, std::abs(static_cast<double>(context.info))),
             relative_error<rank>(context.S_vec.data(), lapack_s.data()), relative_error(internal, expected));
    }

    void run_golub_reinsch_rank_deficient()
    {
        constexpr std::size_t Rows = 8;
        constexpr std::size_t Columns = 4;
        constexpr std::size_t P = 1;
        constexpr std::size_t rank = std::min(Rows, Columns);
        Mat<Rows, Columns> matrix{tensor::uninitialized};
        Mat<Rows, P> seed{tensor::uninitialized};
        Mat<Columns, P> expected{tensor::uninitialized};
        Mat<Rows, P> rhs{tensor::uninitialized};
        Mat<Rows, P> work{};
        Mat<Columns, P> internal{tensor::uninitialized};
        for (std::size_t row = 0; row < Rows; ++row)
        {
            matrix(row, 0) = entry(row, 0, 9) + (row == 0 ? 2.0 : 0.0);
            matrix(row, 1) = entry(row, 1, 9) + (row == 1 ? 2.0 : 0.0);
            matrix(row, 2) = matrix(row, 0) - 2.0 * matrix(row, 1);
            matrix(row, 3) = entry(row, 3, 9) + (row == 3 ? 2.0 : 0.0);
            seed(row, 0) = 0.1 + 0.05 * static_cast<double>(row % 5);
        }
        for (std::size_t row = 0; row < Columns; ++row)
        {
            double value = 0.0;
            for (std::size_t column = 0; column < Rows; ++column)
                value += matrix(column, row) * seed(column, 0);
            expected(row, 0) = value;
        }
        form_rhs(matrix, expected, rhs);

        linalg::Context<linalg::GolubReinsch, Rows, Columns> context;
        context.factorize_from(matrix.data());
        std::copy(rhs.data(), rhs.data() + rhs.count, work.data());
        context.template substitute_inplace<P>(work.data());
        std::copy(work.data(), work.data() + internal.count, internal.data());

        std::array<double, Rows * Columns> factor{};
        std::array<double, Rows * Rows> lapack_u{};
        std::array<double, Columns * Columns> lapack_vt{};
        std::array<double, rank> lapack_s{};
        std::copy(matrix.data(), matrix.data() + matrix.count, factor.data());
        const lapack_int info = LAPACKE_dgesdd(
            LAPACK_ROW_MAJOR, 'A', static_cast<lapack_int>(Rows), static_cast<lapack_int>(Columns), factor.data(),
            static_cast<lapack_int>(Columns), lapack_s.data(), lapack_u.data(), static_cast<lapack_int>(Rows),
            lapack_vt.data(), static_cast<lapack_int>(Columns));
        if (info != 0)
            std::abort();

        const double orthogonality = std::max(orthogonality_error<Rows>(context.U_mat.data()),
                                              orthogonality_error<Columns>(context.Vt_mat.data()));
        const double factor_error = std::max(svd_reconstruction_error(matrix, context), orthogonality);
        emit("golub_reinsch_rank_deficient", std::max(factor_error, std::abs(static_cast<double>(context.info))),
             relative_error<rank>(context.S_vec.data(), lapack_s.data()), relative_error(internal, expected));
    }
} // namespace

int main()
{
    openblas_set_num_threads(1);
    run_doolittle<8, 1>();
    run_doolittle<32, 2>();
    run_doolittle<64, 1>();
    run_doolittle<128, 2>();
    run_cholesky<8, 1>();
    run_cholesky<32, 2>();
    run_cholesky<64, 1>();
    run_cholesky<128, 2>();
    run_bunch_kaufman<8, 1>();
    run_bunch_kaufman<32, 2>();
    run_bunch_kaufman<64, 1>();
    run_bunch_kaufman<128, 2>();
    run_householder<16, 8, 1>();
    run_householder<32, 16, 2>();
    run_householder<64, 32, 1>();
    run_householder<128, 64, 2>();
    run_lu_subnormal_pivot();
    run_cholesky_lower_storage();
    run_cholesky_non_positive();
    run_bunch_kaufman_two_by_two();
    run_bunch_kaufman_one_by_one_swap();
    run_householder_subnormal_reflector();
    run_golub_reinsch<8, 4, 1>();
    run_golub_reinsch<8, 8, 2>();
    run_golub_reinsch<4, 8, 1>();
    run_golub_reinsch<64, 32, 1>();
    run_golub_reinsch<128, 64, 1>();
    run_golub_reinsch_rank_deficient();
    return 0;
}
