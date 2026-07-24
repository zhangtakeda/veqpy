// Smoke coverage for the fixed-size linalg header without LAPACK/BLAS.

#include "linalg.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
    template <std::size_t Rows, std::size_t Columns>
    using Mat = tensor::Matrix<double, Rows, Columns>;

    constexpr auto constexpr_svd()
    {
        Mat<2, 2> matrix{tensor::uninitialized};
        matrix(0, 0) = 3.0;
        matrix(0, 1) = 0.0;
        matrix(1, 0) = 0.0;
        matrix(1, 1) = 2.0;
        return linalg::factorize<linalg::GolubReinsch>(matrix);
    }

    constexpr auto compile_time_svd = constexpr_svd();
    static_assert(compile_time_svd.info == 0);
    static_assert(compile_time_svd.S_vec[0] == 3.0 && compile_time_svd.S_vec[1] == 2.0);

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

    template <std::size_t Rows, std::size_t Columns>
    double relative_error(const Mat<Rows, Columns>& got, const Mat<Rows, Columns>& expected)
    {
        double maximum_error = 0.0;
        double maximum_value = 0.0;
        for (std::size_t index = 0; index < Rows * Columns; ++index)
        {
            maximum_error = std::max(maximum_error, std::abs(got[index] - expected[index]));
            maximum_value = std::max(maximum_value, std::abs(expected[index]));
        }
        return maximum_error / std::max(maximum_value, std::numeric_limits<double>::min());
    }

    bool test_doolittle()
    {
        Mat<3, 3> matrix{4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 3.0};
        Mat<3, 1> expected{0.25, 0.5, 0.75};
        Mat<3, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        const auto context = linalg::factorize<linalg::Doolittle>(matrix);
        context.substitute_inplace(rhs.data());
        return context.info == 0 && relative_error(rhs, expected) < 1.0e-12;
    }

    bool test_cholesky()
    {
        Mat<3, 3> matrix{4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 5.0};
        Mat<3, 1> expected{0.2, 0.3, 0.4};
        Mat<3, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        const auto context = linalg::factorize<linalg::Cholesky>(matrix);
        context.substitute_inplace(rhs.data());
        return context.info == 0 && relative_error(rhs, expected) < 1.0e-12;
    }

    bool test_bunch_kaufman()
    {
        Mat<4, 4> matrix{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.125, 0.0,
                          0.0, 0.125, 2.0, 0.25, 0.0, 0.0, 0.25, 3.0};
        Mat<4, 1> expected{0.25, 0.375, 0.5, 0.625};
        Mat<4, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        const auto context = linalg::factorize<linalg::BunchKaufman>(matrix);
        context.substitute_inplace(rhs.data());
        return context.info == 0 && relative_error(rhs, expected) < 1.0e-12;
    }

    bool test_householder()
    {
        Mat<4, 2> matrix{2.0, 0.5, 0.25, 3.0, -0.5, 0.75, 0.5, -0.25};
        Mat<2, 1> expected{0.25, 0.5};
        Mat<4, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        const auto context = linalg::factorize<linalg::Householder>(matrix);
        context.substitute_inplace(rhs.data());
        Mat<2, 1> got{rhs[0], rhs[1]};
        return context.info == 0 && relative_error(got, expected) < 1.0e-12;
    }

    bool test_golub_reinsch_tall()
    {
        Mat<4, 3> matrix{2.0, 0.25, -0.5, 0.0, 2.5, 0.75, 0.5, -0.25, 2.0, -0.75, 0.5, 0.25};
        Mat<3, 1> expected{0.2, 0.3, 0.4};
        Mat<4, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        const auto context = linalg::factorize<linalg::GolubReinsch>(matrix);
        context.substitute_inplace(rhs.data());
        Mat<3, 1> got{rhs[0], rhs[1], rhs[2]};
        return context.info == 0 && relative_error(got, expected) < 1.0e-11;
    }

    bool test_golub_reinsch_wide()
    {
        Mat<2, 4> matrix{2.0, 0.25, -0.5, 0.75, 0.0, 1.5, 0.5, -0.25};
        Mat<2, 1> seed{0.2, 0.35};
        Mat<4, 1> expected{tensor::uninitialized};
        for (std::size_t row = 0; row < 4; ++row)
        {
            expected(row, 0) = matrix(0, row) * seed(0, 0) + matrix(1, row) * seed(1, 0);
        }
        Mat<2, 1> rhs{tensor::uninitialized};
        form_rhs(matrix, expected, rhs);
        Mat<4, 1> work{};
        std::copy(rhs.data(), rhs.data() + rhs.count, work.data());
        const auto context = linalg::factorize<linalg::GolubReinsch>(matrix);
        context.substitute_inplace(work.data());
        return context.info == 0 && relative_error(work, expected) < 1.0e-11;
    }
} // namespace

int main()
{
    if (!test_doolittle() || !test_cholesky() || !test_bunch_kaufman() || !test_householder() ||
        !test_golub_reinsch_tall() || !test_golub_reinsch_wide())
        return 1;
    std::cout << "fixed-size linalg has no external runtime dependency\n";
    return 0;
}
