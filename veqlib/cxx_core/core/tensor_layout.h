#pragma once

#include "simd.h"
#include "tensor.h"
#include <cstddef>

namespace tensor_layout
{
    using std::size_t;

    using simd::native_lanes_v;

    using tensor::Matrix;
    using tensor::Tensor;

    constexpr size_t ceil_div(size_t numerator, size_t denominator) noexcept
    {
        return (numerator + denominator - 1) / denominator;
    }

    template <size_t K, size_t Rows, size_t Cols, size_t Lanes = native_lanes_v<double>>
    struct MultiMatvecPlan
    {
        static_assert(K >= 1 && K <= 2, "multi-matvec plan supports one or two outputs");
        static_assert(Rows >= 1, "multi-matvec plan requires at least one row");
        static_assert(Cols >= 1, "multi-matvec plan requires at least one column");
        static_assert(Lanes >= 1, "multi-matvec plan requires at least one lane");

        static constexpr size_t output_count = K;
        static constexpr size_t rows         = Rows;
        static constexpr size_t cols         = Cols;
        static constexpr size_t lanes        = Lanes;
        static constexpr size_t row_blocks   = ceil_div(Rows, Lanes);

        Tensor<double, K, row_blocks, Cols, Lanes> coefficients{};

        template <size_t Slot>
        constexpr void load_matrix(const Matrix<double, Rows, Cols>& matrix) noexcept
        {
            static_assert(Slot < K, "multi-matvec matrix slot exceeds output count");

            for (size_t block = 0; block < row_blocks; ++block)
                for (size_t col = 0; col < Cols; ++col)
                    for (size_t lane = 0; lane < Lanes; ++lane)
                    {
                        const size_t row                     = block * Lanes + lane;
                        coefficients(Slot, block, col, lane) = row < Rows ? matrix(row, col) : 0.0;
                    }
        }
    };

    template <size_t Rows, size_t Cols>
    constexpr auto make_matvec_plan(const Matrix<double, Rows, Cols>& matrix) noexcept
    {
        MultiMatvecPlan<1, Rows, Cols> out{};
        out.template load_matrix<0>(matrix);
        return out;
    }

    template <size_t Rows, size_t Cols>
    constexpr auto make_multi_matvec_plan(const Matrix<double, Rows, Cols>& matrix0,
                                          const Matrix<double, Rows, Cols>& matrix1) noexcept
    {
        MultiMatvecPlan<2, Rows, Cols> out{};
        out.template load_matrix<0>(matrix0);
        out.template load_matrix<1>(matrix1);
        return out;
    }

    template <typename GridType>
    struct RadialGridMatvecPlan
    {
        static constexpr auto derivative_accumulator =
            make_multi_matvec_plan(GridType::differentiator, GridType::accumulator);
        static constexpr auto accumulator = make_matvec_plan(GridType::accumulator);
    };
} // namespace tensor_layout
