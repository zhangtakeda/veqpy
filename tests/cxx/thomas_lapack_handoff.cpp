// Runtime coverage for the deliberately remote Thomas -> DGBTRF/DGBTRS handoff.

#include "linalg.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace
{
    constexpr std::size_t bandwidth = 3;
    constexpr std::size_t n         = linalg::detail::thomas_lapack_min_order;

    using BandMatrix = tensor::Matrix<double, bandwidth, n>;
    using Vector     = tensor::Matrix<double, n, 1>;

    static_assert(linalg::detail::uses_runtime_lapack<linalg::Thomas, bandwidth, n>);

    double max_error(const Vector& lhs, const Vector& rhs)
    {
        double result = 0.0;
        for (std::size_t index = 0; index < n; ++index)
            result = std::max(result, std::abs(lhs[index] - rhs[index]));
        return result;
    }
} // namespace

int main()
{
    auto matrix   = std::make_unique<BandMatrix>(tensor::uninitialized);
    auto expected = std::make_unique<Vector>(tensor::uninitialized);
    auto rhs      = std::make_unique<Vector>(tensor::uninitialized);
    auto context  = std::make_unique<linalg::Context<linalg::Thomas, bandwidth, n>>();

    std::fill(matrix->data(), matrix->data() + matrix->count, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        (*matrix)[n + row] = 4.0;
        if (row > 0)
            (*matrix)[2 * n + row - 1] = -0.5;
        if (row + 1 < n)
            (*matrix)[row + 1] = -0.25;
        (*expected)[row] = 0.25 + 0.125 * static_cast<double>(row % 7);
    }

    for (std::size_t row = 0; row < n; ++row)
    {
        (*rhs)[row] = 4.0 * (*expected)[row];
        if (row > 0)
            (*rhs)[row] -= 0.5 * (*expected)[row - 1];
        if (row + 1 < n)
            (*rhs)[row] -= 0.25 * (*expected)[row + 1];
    }

    context->factorize_from(*matrix);
    context->substitute_inplace(rhs->data());
    if (context->info != 0 || max_error(*rhs, *expected) > 1.0e-11)
        return 1;

    std::cout << "Thomas LAPACKE handoff passed\n";
    return 0;
}
