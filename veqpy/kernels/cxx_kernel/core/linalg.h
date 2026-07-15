#pragma once

// Small dense linear-algebra helpers for generated Cxx Kernel artifacts.

#include "math.h"
#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace linalg::detail
{
    using math::sqrt;
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;

    template <typename Policy, size_t N1, size_t N2>
    using Context = typename Policy::template Context<N1, N2>;

    // Test-only escape hatch: the reference driver compares the fixed-size
    // implementation with LAPACKE at sizes that normally dispatch to LAPACK.
    // Production builds do not define this macro.
#if defined(VEQPY_CXX_FORCE_INTERNAL_LINALG)
    inline constexpr bool runtime_lapack_enabled = false;
#else
    inline constexpr bool runtime_lapack_enabled = true;
#endif

    inline constexpr size_t doolittle_lapack_min_size     = 128 * 128;
    inline constexpr size_t cholesky_lapack_min_size      = 16 * 16;
    inline constexpr size_t bunch_kaufman_lapack_min_size = 96 * 96;
    inline constexpr size_t householder_lapack_min_size   = 48 * 48;

    inline constexpr double linear_tolerance = 1.0e-10;

    // Fixed-size, row-major adaptations of the LAPACK unblocked kernels use
    // their exact-zero/safe-scaling semantics.  linear_tolerance remains for
    // the unrelated Thomas helper below.
    inline constexpr double lapack_safe_min = std::numeric_limits<double>::min();
    // DLAMCH('E') is the unit roundoff used by DLARFG (half the C++
    // numeric_limits epsilon on IEEE-754 binary64).
    inline constexpr double lapack_unit_roundoff = std::numeric_limits<double>::epsilon() * 0.5;
    inline constexpr double larfg_safe_min       = lapack_safe_min / lapack_unit_roundoff;

    constexpr bool is_nan(double value) noexcept
    {
        constexpr std::uint64_t exponent_mask = 0x7ff0'0000'0000'0000ULL;
        constexpr std::uint64_t fraction_mask = 0x000f'ffff'ffff'ffffULL;
        const auto bits = std::bit_cast<std::uint64_t>(value);
        return (bits & exponent_mask) == exponent_mask && (bits & fraction_mask) != 0;
    }

    constexpr bool is_negative(double value) noexcept
    {
        return (std::bit_cast<std::uint64_t>(value) & 0x8000'0000'0000'0000ULL) != 0;
    }

    constexpr double stable_norm(const double* values, size_t count, size_t stride = 1)
    {
        double scale = 0.0;
        double ssq   = 1.0;
        for (size_t index = 0; index < count; ++index)
        {
            const double absolute = math::abs(values[index * stride]);
            if (!math::is_finite(absolute))
                return absolute;
            if (absolute == 0.0)
                continue;
            if (scale < absolute)
            {
                const double ratio = scale / absolute;
                ssq                = 1.0 + ssq * ratio * ratio;
                scale              = absolute;
            }
            else
            {
                const double ratio = absolute / scale;
                ssq += ratio * ratio;
            }
        }
        return scale == 0.0 ? 0.0 : scale * sqrt(ssq);
    }

    constexpr double signed_magnitude(double magnitude, double sign_source) noexcept
    {
        return is_negative(sign_source) ? -magnitude : magnitude;
    }

    constexpr void swap_rhs_rows(double* values, size_t lhs, size_t rhs, size_t columns)
    {
        if (lhs == rhs)
            return;
        for (size_t column = 0; column < columns; ++column)
            std::swap(values[lhs * columns + column], values[rhs * columns + column]);
    }

    // Row-major, fixed-storage adaptation of DLARFG.  Keeping this helper
    // scalar and constexpr lets the compiler unroll the known tail length
    // without changing LAPACK's underflow-avoidance and sign rules.
    constexpr void larfg(size_t count, double& alpha, double* tail, size_t stride, double& tau)
    {
        if (count <= 1)
        {
            tau = 0.0;
            return;
        }

        double tail_norm = stable_norm(tail, count - 1, stride);
        if (tail_norm == 0.0)
        {
            tau = 0.0;
            return;
        }

        double beta = -signed_magnitude(math::hypot(alpha, tail_norm), alpha);
        size_t scale_count = 0;
        if (math::abs(beta) < larfg_safe_min)
        {
            const double reciprocal_safe_min = 1.0 / larfg_safe_min;
            do
            {
                ++scale_count;
                for (size_t index = 0; index < count - 1; ++index)
                    tail[index * stride] *= reciprocal_safe_min;
                beta *= reciprocal_safe_min;
                alpha *= reciprocal_safe_min;
            } while (math::abs(beta) < larfg_safe_min && scale_count < 20);

            tail_norm = stable_norm(tail, count - 1, stride);
            beta      = -signed_magnitude(math::hypot(alpha, tail_norm), alpha);
        }

        tau = (beta - alpha) / beta;
        const double reciprocal_alpha_minus_beta = 1.0 / (alpha - beta);
        for (size_t index = 0; index < count - 1; ++index)
            tail[index * stride] *= reciprocal_alpha_minus_beta;
        for (size_t index = 0; index < scale_count; ++index)
            beta *= larfg_safe_min;
        alpha = beta;
    }

    struct BunchKaufman
    {
        template <size_t N1, size_t N2>
        struct Context;

    protected:
        static void lapack_factorize_inplace(int, double*, int, int*);
        static void lapack_substitute_inplace(int, int, const double*, int, const int*, double*, int);
    };

    struct Cholesky
    {
        template <size_t N1, size_t N2>
        struct Context;

    protected:
        static void lapack_factorize_inplace(int, double*, int);
        static void lapack_substitute_inplace(int, int, const double*, int, double*, int);
    };

    struct Doolittle
    {
        template <size_t N1, size_t N2>
        struct Context;

    protected:
        static void lapack_factorize_inplace(int, int, double*, int, int*);
        static void lapack_substitute_inplace(int, int, const double*, int, const int*, double*, int);
    };

    struct GolubReinsch
    {
        template <size_t N1, size_t N2>
        struct Context;

    protected:
        static void lapack_factorize_inplace(int, int, const double*, double*, double*, double*);
        static void lapack_substitute_inplace(int, int, const double*, const double*, const double*, double*, int);
    };

    struct Householder
    {
        template <size_t N1, size_t N2>
        struct Context;

    protected:
        static void lapack_factorize_inplace(int, int, double*, int, double*);
        static void lapack_substitute_inplace(int, int, int, const double*, int, const double*, double*, int);
    };

    struct Thomas
    {
        template <size_t Bandwidth, size_t N>
        struct Context;
    };

    template <typename Policy = Doolittle, size_t N1, size_t N2>
    constexpr Context<Policy, N1, N2>& factorize_into(Context<Policy, N1, N2>&              context,
                                                      const tensor::Matrix<double, N1, N2>& matrix)
    {
        if constexpr (std::is_same_v<Policy, Thomas>)
            context.factorize_from(matrix);
        else
            context.factorize_from(matrix.data());
        return context;
    }

    template <typename Policy = Doolittle, size_t N1, size_t N2>
    constexpr Context<Policy, N1, N2> factorize(const tensor::Matrix<double, N1, N2>& matrix)
    {
        Context<Policy, N1, N2> context;
        factorize_into<Policy>(context, matrix);
        return context;
    }

    template <typename Policy, size_t N1, size_t N2, size_t P>
    using Rhs = tensor::Matrix<double, std::is_same_v<Policy, Thomas> ? N2 : N1, P>;

    template <typename Policy = Doolittle, size_t N1, size_t N2, size_t P>
    constexpr void solve_into(tensor::Matrix<double, N2, P>&        solution,
                              const tensor::Matrix<double, N1, N2>& matrix,
                              const Rhs<Policy, N1, N2, P>&         rhs)
    {
        if constexpr (std::is_same_v<Policy, Thomas>)
        {
            std::copy(rhs.data(), rhs.data() + rhs.count, solution.data());

            const auto context = factorize<Policy>(matrix);
            context.template substitute_inplace<P>(solution.data());
        }
        else if constexpr (N1 > N2)
        {
            tensor::Matrix<double, N1, P> work{tensor::uninitialized};
            std::copy(rhs.data(), rhs.data() + N1 * P, work.data());

            const auto context = factorize<Policy>(matrix);
            context.template substitute_inplace<P>(work.data());
            std::copy(work.data(), work.data() + N2 * P, solution.data());
        }
        else
        {
            std::copy(rhs.data(), rhs.data() + N1 * P, solution.data());

            const auto context = factorize<Policy>(matrix);
            context.template substitute_inplace<P>(solution.data());
        }
    }

    template <typename Policy = Doolittle, size_t N1, size_t N2, size_t P>
    constexpr auto solve(const tensor::Matrix<double, N1, N2>& matrix, const Rhs<Policy, N1, N2, P>& rhs)
    {
        tensor::Matrix<double, N2, P> solution{tensor::uninitialized};
        solve_into<Policy>(solution, matrix, rhs);
        return solution;
    }

    template <size_t N1, size_t N2, size_t N3>
    constexpr void matmul_into(tensor::Matrix<double, N1, N3>&       out,
                               const tensor::Matrix<double, N1, N2>& lhs,
                               const tensor::Matrix<double, N2, N3>& rhs)
    {
        if (out.data() == lhs.data() || out.data() == rhs.data())
        {
            tensor::Matrix<double, N1, N3> tmp{tensor::uninitialized};
            matmul_into(tmp, lhs, rhs);
            std::copy(tmp.data(), tmp.data() + tmp.count, out.data());
            return;
        }

        const auto* lhs_data = lhs.data();
        const auto* rhs_data = rhs.data();
        auto*       out_data = out.data();

        for (size_t i = 0; i < N1; ++i)
            for (size_t j = 0; j < N3; ++j)
            {
                double total = 0.0;
                for (size_t k = 0; k < N2; ++k)
                    total += lhs_data[i * N2 + k] * rhs_data[k * N3 + j];
                out_data[i * N3 + j] = total;
            }
    }

    template <size_t N1, size_t N2, size_t N3>
    constexpr tensor::Matrix<double, N1, N3> matmul(const tensor::Matrix<double, N1, N2>& lhs,
                                                    const tensor::Matrix<double, N2, N3>& rhs)
    {
        tensor::Matrix<double, N1, N3> out{tensor::uninitialized};
        matmul_into(out, lhs, rhs);
        return out;
    }

    template <size_t N1, size_t N2>
    constexpr void transpose_into(tensor::Matrix<double, N2, N1>& out, const tensor::Matrix<double, N1, N2>& in)
    {
        if constexpr (N1 == N2)
        {
            if (out.data() == in.data())
            {
                auto* out_data = out.data();
                for (size_t i = 0; i < N1; ++i)
                    for (size_t j = i + 1; j < N2; ++j)
                        std::swap(out_data[i * N2 + j], out_data[j * N2 + i]);
                return;
            }
        }

        auto*       out_data = out.data();
        const auto* in_data  = in.data();
        for (size_t i = 0; i < N1; ++i)
            for (size_t j = 0; j < N2; ++j)
                out_data[j * N1 + i] = in_data[i * N2 + j];
    }

    template <size_t N1, size_t N3>
    constexpr tensor::Matrix<double, N3, N1> transpose(const tensor::Matrix<double, N1, N3>& in)
    {
        tensor::Matrix<double, N3, N1> out{tensor::uninitialized};
        transpose_into(out, in);
        return out;
    }

    template <size_t N1, size_t N2>
    struct BunchKaufman::Context
    {
        static_assert(N1 == N2, "BunchKaufman only supports square matrices");

        // (1 + sqrt(17)) / 8, the Bunch--Kaufman threshold in DSYTF2.
        static constexpr double alpha = 0.6403882032022076;

        Matrix<double, N1, N1> LDLT_mat{tensor::uninitialized};
        Vector<int, N1>        pivot_vec{tensor::uninitialized};
        int                     info{0};

        constexpr void factorize_from(const double* A)
        {
            double* L    = LDLT_mat.data();
            int*    ipiv = pivot_vec.data();
            info         = 0;
            std::copy(A, A + N1 * N1, L);

            if constexpr (runtime_lapack_enabled && N1 * N1 >= bunch_kaufman_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, L, N1, ipiv);
                    return;
                }
            }

            // DSYTF2('L') translated to row-major fixed storage.  It only
            // reads and overwrites the lower triangle, exactly like LAPACK;
            // the full Matrix remains solely a convenient fixed-size owner.
            for (size_t k = 0; k < N1;)
            {
                size_t kstep = 1;
                size_t kp    = k;
                const double absakk = math::abs(L[k * N1 + k]);

                size_t imax   = k;
                double colmax = 0.0;
                if (k + 1 < N1)
                {
                    for (size_t row = k + 1; row < N1; ++row)
                    {
                        const double absolute = math::abs(L[row * N1 + k]);
                        if (absolute > colmax)
                        {
                            colmax = absolute;
                            imax   = row;
                        }
                    }
                }

                const bool singular_column = math::max(absakk, colmax) == 0.0 || is_nan(absakk);
                if (singular_column)
                {
                    if (info == 0)
                        info = static_cast<int>(k + 1);
                }
                else if (absakk >= alpha * colmax)
                {
                    kp = k;
                }
                else
                {
                    double rowmax = 0.0;
                    for (size_t column = k; column < imax; ++column)
                    {
                        const double absolute = math::abs(L[imax * N1 + column]);
                        if (absolute > rowmax)
                            rowmax = absolute;
                    }

                    for (size_t row = imax + 1; row < N1; ++row)
                    {
                        const double absolute = math::abs(L[row * N1 + imax]);
                        if (absolute > rowmax)
                            rowmax = absolute;
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax))
                    {
                        kp = k;
                    }
                    else if (math::abs(L[imax * N1 + imax]) >= alpha * rowmax)
                    {
                        kp = imax;
                    }
                    else
                    {
                        kp    = imax;
                        kstep = 2;
                    }
                }

                if (!singular_column)
                {
                    const size_t kk = k + kstep - 1;
                    if (kp != kk)
                    {
                        for (size_t row = kp + 1; row < N1; ++row)
                            std::swap(L[row * N1 + kk], L[row * N1 + kp]);
                        for (size_t row = kk + 1; row < kp; ++row)
                            std::swap(L[row * N1 + kk], L[kp * N1 + row]);
                        std::swap(L[kk * N1 + kk], L[kp * N1 + kp]);
                        if (kstep == 2)
                            std::swap(L[(k + 1) * N1 + k], L[kp * N1 + k]);
                    }

                    if (kstep == 1)
                    {
                        if (k + 1 < N1)
                        {
                            const double d11 = 1.0 / L[k * N1 + k];
                            for (size_t column = k + 1; column < N1; ++column)
                                for (size_t row = column; row < N1; ++row)
                                    L[row * N1 + column] -= L[row * N1 + k] * L[column * N1 + k] * d11;
                            for (size_t row = k + 1; row < N1; ++row)
                                L[row * N1 + k] *= d11;
                        }
                    }
                    else if (k + 2 < N1)
                    {
                        double d21 = L[(k + 1) * N1 + k];
                        const double d11 = L[(k + 1) * N1 + (k + 1)] / d21;
                        const double d22 = L[k * N1 + k] / d21;
                        const double scale = 1.0 / (d11 * d22 - 1.0);
                        d21 = scale / d21;

                        for (size_t column = k + 2; column < N1; ++column)
                        {
                            const double wk = d21 * (d11 * L[column * N1 + k] - L[column * N1 + (k + 1)]);
                            const double wkp1 =
                                d21 * (d22 * L[column * N1 + (k + 1)] - L[column * N1 + k]);
                            for (size_t row = column; row < N1; ++row)
                                L[row * N1 + column] -=
                                    L[row * N1 + k] * wk + L[row * N1 + (k + 1)] * wkp1;
                            L[column * N1 + k]       = wk;
                            L[column * N1 + (k + 1)] = wkp1;
                        }
                    }
                }

                if (kstep == 1)
                    ipiv[k] = static_cast<int>(kp + 1);
                else
                {
                    ipiv[k]     = -static_cast<int>(kp + 1);
                    ipiv[k + 1] = -static_cast<int>(kp + 1);
                }

                k += kstep;
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* L    = LDLT_mat.data();
            const int*    ipiv = pivot_vec.data();

            if constexpr (runtime_lapack_enabled && N1 * N1 >= bunch_kaufman_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, L, N1, ipiv, x, P);
                    return;
                }
            }

            // DSYTRS('L'), sharing LAPACK's one-based signed IPIV encoding
            // with the runtime path.  Positive entries denote 1x1 blocks;
            // equal negative entries denote a 2x2 block.
            for (size_t k = 0; k < N1;)
            {
                if (ipiv[k] > 0)
                {
                    const size_t kp = static_cast<size_t>(ipiv[k] - 1);
                    swap_rhs_rows(x, k, kp, P);
                    for (size_t row = k + 1; row < N1; ++row)
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            x[row * P + rhs] -= L[row * N1 + k] * x[k * P + rhs];
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        x[k * P + rhs] /= L[k * N1 + k];
                    ++k;
                }
                else
                {
                    const size_t kp = static_cast<size_t>(-ipiv[k] - 1);
                    swap_rhs_rows(x, k + 1, kp, P);
                    for (size_t row = k + 2; row < N1; ++row)
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            x[row * P + rhs] -= L[row * N1 + k] * x[k * P + rhs];
                    for (size_t row = k + 2; row < N1; ++row)
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            x[row * P + rhs] -= L[row * N1 + (k + 1)] * x[(k + 1) * P + rhs];

                    const double akm1k = L[(k + 1) * N1 + k];
                    const double akm1  = L[k * N1 + k] / akm1k;
                    const double ak    = L[(k + 1) * N1 + (k + 1)] / akm1k;
                    const double denom = akm1 * ak - 1.0;
                    for (size_t rhs = 0; rhs < P; ++rhs)
                    {
                        const double bkm1 = x[k * P + rhs] / akm1k;
                        const double bk   = x[(k + 1) * P + rhs] / akm1k;
                        x[k * P + rhs]       = (ak * bkm1 - bk) / denom;
                        x[(k + 1) * P + rhs] = (akm1 * bk - bkm1) / denom;
                    }
                    k += 2;
                }
            }

            for (size_t remaining = N1; remaining > 0;)
            {
                const size_t k = remaining - 1;
                if (ipiv[k] > 0)
                {
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        for (size_t row = k + 1; row < N1; ++row)
                            x[k * P + rhs] -= L[row * N1 + k] * x[row * P + rhs];
                    const size_t kp = static_cast<size_t>(ipiv[k] - 1);
                    swap_rhs_rows(x, k, kp, P);
                    remaining = k;
                }
                else
                {
                    const size_t km1 = k - 1;
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        for (size_t row = k + 1; row < N1; ++row)
                            x[k * P + rhs] -= L[row * N1 + k] * x[row * P + rhs];
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        for (size_t row = k + 1; row < N1; ++row)
                            x[km1 * P + rhs] -= L[row * N1 + km1] * x[row * P + rhs];
                    const size_t kp = static_cast<size_t>(-ipiv[k] - 1);
                    swap_rhs_rows(x, k, kp, P);
                    remaining = km1;
                }
            }
        }
    };

    template <size_t N1, size_t N2>
    struct Cholesky::Context
    {
        static_assert(N1 == N2, "cholesky only supports square matrices");

        Matrix<double, N1, N1> LLT_mat{tensor::uninitialized};
        int                    info{0};

        constexpr void factorize_from(const double* A)
        {
            double* L = LLT_mat.data();
            info      = 0;
            std::copy(A, A + N1 * N1, L);

            if constexpr (runtime_lapack_enabled && N1 * N1 >= cholesky_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, L, N1);
                    return;
                }
            }

            // DPOTF2('L') consumes and overwrites only the lower triangle;
            // it deliberately does not validate or symmetrize the upper one.
            for (size_t j = 0; j < N1; ++j)
            {
                double ajj = L[j * N1 + j];
                for (size_t k = 0; k < j; ++k)
                    ajj -= L[j * N1 + k] * L[j * N1 + k];

                if (ajj <= 0.0 || is_nan(ajj))
                {
                    L[j * N1 + j] = ajj;
                    info            = static_cast<int>(j + 1);
                    return;
                }

                ajj              = sqrt(ajj);
                L[j * N1 + j]    = ajj;
                const double inv = 1.0 / ajj;
                for (size_t row = j + 1; row < N1; ++row)
                {
                    double value = L[row * N1 + j];
                    for (size_t k = 0; k < j; ++k)
                        value -= L[row * N1 + k] * L[j * N1 + k];
                    L[row * N1 + j] = value * inv;
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* L = LLT_mat.data();

            if constexpr (runtime_lapack_enabled && N1 * N1 >= cholesky_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, L, N1, x, P);
                    return;
                }
            }

            for (size_t i = 0; i < N1; ++i)
            {
                for (size_t j = 0; j < P; ++j)
                {
                    for (size_t k = 0; k < i; ++k)
                        x[i * P + j] -= L[i * N1 + k] * x[k * P + j];

                    x[i * P + j] /= L[i * N1 + i];
                }
            }

            for (size_t ii = N1; ii > 0; --ii)
            {
                const size_t i = ii - 1;
                for (size_t j = 0; j < P; ++j)
                {
                    for (size_t k = i + 1; k < N1; ++k)
                        x[i * P + j] -= L[k * N1 + i] * x[k * P + j];

                    x[i * P + j] /= L[i * N1 + i];
                }
            }
        }
    };

    template <size_t N1, size_t N2>
    struct Doolittle::Context
    {
        static_assert(N1 == N2, "Doolittle only supports square matrices");

        Matrix<double, N1, N1> LU_mat{tensor::uninitialized};
        Vector<int, N1>        pivot_vec{tensor::uninitialized};
        int                     info{0};

        constexpr void factorize_from(const double* A)
        {
            double* LU   = LU_mat.data();
            int*    ipiv = pivot_vec.data();
            info         = 0;
            std::copy(A, A + N1 * N1, LU);

            if constexpr (runtime_lapack_enabled && N1 * N1 >= doolittle_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, N1, LU, N1, ipiv);
                    return;
                }
            }

            // DGETF2 with a row-major storage interpretation.  At fixed
            // dimensions the loops are compile-time bounded, while pivots,
            // safe-min scaling, and the singular-info convention stay the
            // same as the LAPACK reference kernel.
            for (size_t k = 0; k < N1; ++k)
            {
                size_t pivot_row = k;
                double max_norm  = math::abs(LU[k * N1 + k]);

                for (size_t i = k + 1; i < N1; ++i)
                {
                    const double curr_norm = math::abs(LU[i * N1 + k]);
                    if (curr_norm > max_norm)
                    {
                        max_norm  = curr_norm;
                        pivot_row = i;
                    }
                }

                ipiv[k] = static_cast<int>(pivot_row);
                if (LU[pivot_row * N1 + k] != 0.0)
                {
                    if (pivot_row != k)
                        std::swap_ranges(LU + k * N1, LU + (k + 1) * N1, LU + pivot_row * N1);

                    if (k + 1 < N1)
                    {
                        const double pivot = LU[k * N1 + k];
                        if (math::abs(pivot) >= lapack_safe_min)
                        {
                            const double reciprocal_pivot = 1.0 / pivot;
                            for (size_t row = k + 1; row < N1; ++row)
                                LU[row * N1 + k] *= reciprocal_pivot;
                        }
                        else
                        {
                            for (size_t row = k + 1; row < N1; ++row)
                                LU[row * N1 + k] /= pivot;
                        }
                    }
                }
                else if (info == 0)
                {
                    info = static_cast<int>(k + 1);
                }

                if (k + 1 < N1)
                    for (size_t column = k + 1; column < N1; ++column)
                        for (size_t row = k + 1; row < N1; ++row)
                            LU[row * N1 + column] -= LU[row * N1 + k] * LU[k * N1 + column];
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* LU   = LU_mat.data();
            const int*    ipiv = pivot_vec.data();

            if constexpr (runtime_lapack_enabled && N1 * N1 >= doolittle_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, LU, N1, ipiv, x, P);
                    return;
                }
            }

            for (size_t k = 0; k < N1; ++k)
            {
                const int pivot_row = ipiv[k];
                if (pivot_row != static_cast<int>(k))
                    swap_rhs_rows(x, k, static_cast<size_t>(pivot_row), P);
            }

            for (size_t i = 1; i < N1; ++i)
                for (size_t j = 0; j < P; ++j)
                    for (size_t k = 0; k < i; ++k)
                        x[i * P + j] -= LU[i * N1 + k] * x[k * P + j];

            for (size_t i = N1; i-- > 0;)
            {
                for (size_t j = 0; j < P; ++j)
                {
                    for (size_t k = i + 1; k < N1; ++k)
                        x[i * P + j] -= LU[i * N1 + k] * x[k * P + j];

                    x[i * P + j] /= LU[i * N1 + i];
                }
            }
        }
    };

    template <size_t N1, size_t N2>
    struct GolubReinsch::Context
    {
        Matrix<double, N1, N1>           U_mat{tensor::uninitialized};
        Matrix<double, N2, N2>           Vt_mat{tensor::uninitialized};
        Vector<double, std::min(N1, N2)> S_vec{tensor::uninitialized};

        constexpr void factorize_from(const double* A)
        {
            double* U  = U_mat.data();
            double* Vt = Vt_mat.data();
            double* S  = S_vec.data();

            if (std::is_constant_evaluated())
                throw "GolubReinsch requires runtime LAPACK";
            lapack_factorize_inplace(N1, N2, A, U, S, Vt);
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* U  = U_mat.data();
            const double* Vt = Vt_mat.data();
            const double* S  = S_vec.data();

            if (std::is_constant_evaluated())
                throw "GolubReinsch requires runtime LAPACK";
            lapack_substitute_inplace(N1, N2, U, S, Vt, x, P);
        }
    };

    template <size_t N1, size_t N2>
    struct Householder::Context
    {
        static_assert(N1 >= N2, "QR only supports matrices with at least as many N1 as columns");

        Matrix<double, N1, N2> QR_mat{tensor::uninitialized};
        Vector<double, N2>     tau_vec{tensor::uninitialized};
        int                     info{0};

        constexpr void factorize_from(const double* A)
        {
            double* QR  = QR_mat.data();
            double* tau = tau_vec.data();
            info        = 0;
            std::copy(A, A + N1 * N2, QR);

            if constexpr (runtime_lapack_enabled && N1 * N2 >= householder_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, N2, QR, N2, tau);
                    return;
                }
            }

            // DGEQR2: the compact WY-free representation is already the
            // natural fixed-array form, so only DLARFG/DLARF's scalar logic
            // is needed here.
            for (size_t k = 0; k < N2; ++k)
            {
                double* tail = k + 1 < N1 ? QR + (k + 1) * N2 + k : QR + k * N2 + k;
                larfg(N1 - k, QR[k * N2 + k], tail, N2, tau[k]);

                if (k + 1 < N2 && tau[k] != 0.0)
                {
                    for (size_t column = k + 1; column < N2; ++column)
                    {
                        double dot = QR[k * N2 + column];
                        for (size_t row = k + 1; row < N1; ++row)
                            dot += QR[row * N2 + k] * QR[row * N2 + column];

                        dot *= tau[k];
                        QR[k * N2 + column] -= dot;
                        for (size_t row = k + 1; row < N1; ++row)
                            QR[row * N2 + column] -= QR[row * N2 + k] * dot;
                    }
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* QR  = QR_mat.data();
            const double* tau = tau_vec.data();

            if constexpr (runtime_lapack_enabled && N1 * N2 >= householder_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, N2, P, QR, N2, tau, x, P);
                    return;
                }
            }

            for (size_t k = 0; k < N2; ++k)
            {
                if (tau[k] == 0.0)
                    continue;

                for (size_t j = 0; j < P; ++j)
                {
                    double vTx_j = x[k * P + j];
                    for (size_t i = k + 1; i < N1; ++i)
                        vTx_j += QR[i * N2 + k] * x[i * P + j];

                    vTx_j *= tau[k];
                    x[k * P + j] -= vTx_j;
                    for (size_t i = k + 1; i < N1; ++i)
                        x[i * P + j] -= QR[i * N2 + k] * vTx_j;
                }
            }

            for (size_t j = 0; j < P; ++j)
            {
                for (size_t i = N2; i > 0; --i)
                {
                    double sum = x[(i - 1) * P + j];
                    for (size_t k = i; k < N2; ++k)
                        sum -= QR[(i - 1) * N2 + k] * x[k * P + j];

                    x[(i - 1) * P + j] = sum / QR[(i - 1) * N2 + (i - 1)];
                }
            }
        }
    };

    template <size_t Bandwidth, size_t N>
    struct Thomas::Context
    {
        static_assert(Bandwidth % 2 == 1, "Thomas band width must be odd");
        static_assert(N >= Bandwidth, "Thomas matrix must have at least Bandwidth rows");

        static constexpr size_t radius = Bandwidth / 2;

        Matrix<double, Bandwidth, N> LU_mat{tensor::uninitialized};

        static constexpr bool in_band(size_t row, size_t col) noexcept
        {
            return row + radius >= col && col + radius >= row;
        }

        static constexpr size_t band_index(size_t row, size_t col) noexcept { return radius + row - col; }

        static constexpr double get(const Matrix<double, Bandwidth, N>& matrix, size_t row, size_t col) noexcept
        {
            if (!in_band(row, col))
                return 0.0;
            return matrix[band_index(row, col) * N + col];
        }

        static constexpr void set(Matrix<double, Bandwidth, N>& matrix, size_t row, size_t col, double value) noexcept
        {
            assert(in_band(row, col));
            matrix[band_index(row, col) * N + col] = value;
        }

        constexpr void factorize_from(const Matrix<double, Bandwidth, N>& matrix)
        {
            std::copy(matrix.data(), matrix.data() + matrix.count, LU_mat.data());

            for (size_t pivot = 0; pivot + 1 < N; ++pivot)
            {
                const double pivot_value = get(LU_mat, pivot, pivot);
                assert(std::abs(pivot_value) >= linear_tolerance);

                const size_t row_stop = std::min(N, pivot + radius + 1);
                const size_t col_stop = std::min(N, pivot + radius + 1);
                for (size_t row = pivot + 1; row < row_stop; ++row)
                {
                    const double factor = get(LU_mat, row, pivot) / pivot_value;
                    set(LU_mat, row, pivot, factor);

                    for (size_t col = pivot + 1; col < col_stop; ++col)
                    {
                        if (!in_band(row, col))
                            continue;
                        set(LU_mat, row, col, get(LU_mat, row, col) - factor * get(LU_mat, pivot, col));
                    }
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const auto& LU = LU_mat;

            for (size_t row = 0; row < N; ++row)
            {
                const size_t col_start = row > radius ? row - radius : 0;
                for (size_t col = col_start; col < row; ++col)
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        x[row * P + rhs] -= get(LU, row, col) * x[col * P + rhs];
            }

            for (size_t rr = N; rr > 0; --rr)
            {
                const size_t row      = rr - 1;
                const size_t col_stop = std::min(N, row + radius + 1);
                for (size_t col = row + 1; col < col_stop; ++col)
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        x[row * P + rhs] -= get(LU, row, col) * x[col * P + rhs];

                const double pivot_value = get(LU, row, row);
                assert(std::abs(pivot_value) >= linear_tolerance);
                for (size_t rhs = 0; rhs < P; ++rhs)
                    x[row * P + rhs] /= pivot_value;
            }
        }
    };
} // namespace linalg::detail

namespace linalg
{
    using detail::BunchKaufman;
    using detail::Cholesky;
    using detail::Doolittle;
    using detail::GolubReinsch;
    using detail::Householder;
    using detail::Thomas;

    using detail::Context;
    using detail::factorize;
    using detail::factorize_into;
    using detail::matmul;
    using detail::matmul_into;
    using detail::solve;
    using detail::solve_into;
    using detail::transpose;
    using detail::transpose_into;
} // namespace linalg
