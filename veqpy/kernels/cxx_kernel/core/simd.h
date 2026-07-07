#pragma once

// SIMD lane traits and vector utilities for generated Cxx Kernel artifacts.

#include <cstddef>
#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace simd
{
    using std::size_t;

    inline constexpr size_t f64x4_lanes = 4;

    template <typename T>
    inline constexpr size_t native_lanes_v = 1;

    template <>
    inline constexpr size_t native_lanes_v<double> = f64x4_lanes;

#if defined(__AVX2__)
    inline void
    matvec_f64x4_lanes_into(double* lanes, const double* coefficients, const double* input, size_t cols) noexcept
    {
        __m256d total = _mm256_setzero_pd();
        for (size_t col = 0; col < cols; ++col)
        {
            const __m256d value  = _mm256_broadcast_sd(input + col);
            const __m256d matrix = _mm256_load_pd(coefficients + col * f64x4_lanes);
    #if defined(__FMA__)
            total = _mm256_fmadd_pd(matrix, value, total);
    #else
            total = _mm256_add_pd(total, _mm256_mul_pd(matrix, value));
    #endif
        }
        _mm256_store_pd(lanes, total);
    }

    inline void multi_matvec2_f64x4_lanes_into(double*       lanes0,
                                               double*       lanes1,
                                               const double* coefficients0,
                                               const double* coefficients1,
                                               const double* input,
                                               size_t        cols) noexcept
    {
        __m256d total0 = _mm256_setzero_pd();
        __m256d total1 = _mm256_setzero_pd();
        for (size_t col = 0; col < cols; ++col)
        {
            const __m256d value   = _mm256_broadcast_sd(input + col);
            const __m256d matrix0 = _mm256_load_pd(coefficients0 + col * f64x4_lanes);
            const __m256d matrix1 = _mm256_load_pd(coefficients1 + col * f64x4_lanes);
    #if defined(__FMA__)
            total0 = _mm256_fmadd_pd(matrix0, value, total0);
            total1 = _mm256_fmadd_pd(matrix1, value, total1);
    #else
            total0 = _mm256_add_pd(total0, _mm256_mul_pd(matrix0, value));
            total1 = _mm256_add_pd(total1, _mm256_mul_pd(matrix1, value));
    #endif
        }
        _mm256_store_pd(lanes0, total0);
        _mm256_store_pd(lanes1, total1);
    }
#endif
} // namespace simd
