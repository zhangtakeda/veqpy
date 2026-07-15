// Differential cases for the fixed-size VEQ linalg policies.
//
// The driver intentionally forces the in-tree templates and compares their
// factorization + solve result with the row-major LAPACKE entry points used by
// the legacy runtime path.  The matrices are deterministic and well
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

    constexpr double entry(std::size_t row, std::size_t column, std::size_t salt)
    {
        const auto value = (row * 37 + column * 17 + salt * 13 + 11) % 41;
        return (static_cast<double>(value) - 20.0) / 41.0;
    }

    template <std::size_t N, std::size_t P>
    void form_rhs(const Mat<N, N>& matrix, const Mat<N, P>& expected, Mat<N, P>& rhs)
    {
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t rhs_column = 0; rhs_column < P; ++rhs_column)
            {
                double value = 0.0;
                for (std::size_t column = 0; column < N; ++column)
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
    return 0;
}
