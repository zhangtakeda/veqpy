#pragma once

#include "simd.h"
#include "tensor.h"
#include "tensor_layout.h"
#include <cstddef>
#include <type_traits>

namespace tensor_kernels
{
    using std::size_t;

    using simd::f64x4_lanes;
#if defined(__AVX2__)
    using simd::matvec_f64x4_lanes_into;
    using simd::multi_matvec2_f64x4_lanes_into;
#endif

    using tensor::Matrix;
    using tensor::Vector;
    using tensor_layout::MultiMatvecPlan;

    template <size_t Rows, size_t Cols>
    constexpr void matvec_into(Vector<double, Rows>&       out,
                               const Matrix<double, Rows, Cols>& matrix,
                               const Vector<double, Cols>& values) noexcept
    {
        for (size_t row = 0; row < Rows; ++row)
        {
            double total = 0.0;
            for (size_t col = 0; col < Cols; ++col)
                total += matrix(row, col) * values[col];
            out[row] = total;
        }
    }

    template <size_t Rows, size_t Cols>
    constexpr void multi_matvec_into(Vector<double, Rows>&       out0,
                                     Vector<double, Rows>&       out1,
                                     const Matrix<double, Rows, Cols>& matrix0,
                                     const Matrix<double, Rows, Cols>& matrix1,
                                     const Vector<double, Cols>& values) noexcept
    {
        for (size_t row = 0; row < Rows; ++row)
        {
            double total0 = 0.0;
            double total1 = 0.0;
            for (size_t col = 0; col < Cols; ++col)
            {
                const double value = values[col];
                total0 += matrix0(row, col) * value;
                total1 += matrix1(row, col) * value;
            }
            out0[row] = total0;
            out1[row] = total1;
        }
    }

    namespace detail
    {
        template <size_t Slot, size_t K, size_t Rows, size_t Cols, size_t Lanes>
        constexpr void plan_slot_matvec_scalar_into(
            Vector<double, Rows>&                              out,
            const MultiMatvecPlan<K, Rows, Cols, Lanes>& plan,
            const Vector<double, Cols>&                  values) noexcept
        {
            static_assert(Slot < K, "multi-matvec matrix slot exceeds output count");

            for (size_t block = 0; block < plan.row_blocks; ++block)
            {
                double totals[Lanes]{};
                for (size_t col = 0; col < Cols; ++col)
                {
                    const double value = values[col];
                    for (size_t lane = 0; lane < Lanes; ++lane)
                        totals[lane] += plan.coefficients(Slot, block, col, lane) * value;
                }
                for (size_t lane = 0; lane < Lanes; ++lane)
                {
                    const size_t row = block * Lanes + lane;
                    if (row < Rows)
                        out[row] = totals[lane];
                }
            }
        }

#if defined(__AVX2__)
        template <size_t Rows, size_t Cols>
        inline void plan_matvec_f64x4_into(
            Vector<double, Rows>&                                        out,
            const MultiMatvecPlan<1, Rows, Cols, f64x4_lanes>& plan,
            const Vector<double, Cols>&                            values) noexcept
        {
            for (size_t block = 0; block < plan.row_blocks; ++block)
            {
                alignas(32) double lanes[f64x4_lanes];
                matvec_f64x4_lanes_into(
                    lanes, &plan.coefficients(0, block, 0, 0), values.data(), Cols);
                for (size_t lane = 0; lane < f64x4_lanes; ++lane)
                {
                    const size_t row = block * f64x4_lanes + lane;
                    if (row < Rows)
                        out[row] = lanes[lane];
                }
            }
        }

        template <size_t Rows, size_t Cols>
        inline void plan_multi_matvec2_f64x4_into(
            Vector<double, Rows>&                                        out0,
            Vector<double, Rows>&                                        out1,
            const MultiMatvecPlan<2, Rows, Cols, f64x4_lanes>& plan,
            const Vector<double, Cols>&                            values) noexcept
        {
            for (size_t block = 0; block < plan.row_blocks; ++block)
            {
                alignas(32) double lanes0[f64x4_lanes];
                alignas(32) double lanes1[f64x4_lanes];
                multi_matvec2_f64x4_lanes_into(
                    lanes0,
                    lanes1,
                    &plan.coefficients(0, block, 0, 0),
                    &plan.coefficients(1, block, 0, 0),
                    values.data(),
                    Cols);
                for (size_t lane = 0; lane < f64x4_lanes; ++lane)
                {
                    const size_t row = block * f64x4_lanes + lane;
                    if (row < Rows)
                    {
                        out0[row] = lanes0[lane];
                        out1[row] = lanes1[lane];
                    }
                }
            }
        }
#endif
    } // namespace detail

    template <size_t Rows, size_t Cols, size_t Lanes>
    constexpr void multi_matvec_into(Vector<double, Rows>&                         out,
                                     const MultiMatvecPlan<1, Rows, Cols, Lanes>& plan,
                                     const Vector<double, Cols>&                  values) noexcept
    {
#if defined(__AVX2__)
        if (!std::is_constant_evaluated())
        {
            if constexpr (Lanes == f64x4_lanes)
            {
                detail::plan_matvec_f64x4_into(out, plan, values);
                return;
            }
        }
#endif
        detail::plan_slot_matvec_scalar_into<0>(out, plan, values);
    }

    template <size_t Rows, size_t Cols, size_t Lanes>
    constexpr void matvec_into(Vector<double, Rows>&                         out,
                               const MultiMatvecPlan<1, Rows, Cols, Lanes>& plan,
                               const Vector<double, Cols>&                  values) noexcept
    {
        multi_matvec_into(out, plan, values);
    }

    template <size_t Rows, size_t Cols, size_t Lanes>
    constexpr void multi_matvec_into(Vector<double, Rows>&                         out0,
                                     Vector<double, Rows>&                         out1,
                                     const MultiMatvecPlan<2, Rows, Cols, Lanes>& plan,
                                     const Vector<double, Cols>&                  values) noexcept
    {
#if defined(__AVX2__)
        if (!std::is_constant_evaluated())
        {
            if constexpr (Lanes == f64x4_lanes)
            {
                detail::plan_multi_matvec2_f64x4_into(out0, out1, plan, values);
                return;
            }
        }
#endif
        detail::plan_slot_matvec_scalar_into<0>(out0, plan, values);
        detail::plan_slot_matvec_scalar_into<1>(out1, plan, values);
    }
} // namespace tensor_kernels
