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

    // Keep a fully constexpr, fixed-storage route for tests and for callers
    // that require constant evaluation.  Normal generated kernels select the
    // blocked LAPACKE routines only once their fixed order reaches the
    // measured crossover for that policy.
#if defined(VEQPY_CXX_FORCE_INTERNAL_LINALG)
    inline constexpr bool runtime_lapack_enabled = false;
#else
    inline constexpr bool runtime_lapack_enabled = true;
#endif

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

    template <size_t Rows, size_t Columns>
    constexpr void set_identity(double* matrix)
    {
        static_assert(Rows == Columns, "identity requires a square matrix");
        for (size_t row = 0; row < Rows; ++row)
            for (size_t column = 0; column < Columns; ++column)
                matrix[row * Columns + column] = row == column ? 1.0 : 0.0;
    }

    template <size_t Rows, size_t Columns>
    constexpr void rotate_columns(double* matrix, size_t left, size_t right, double cosine, double sine)
    {
        for (size_t row = 0; row < Rows; ++row)
        {
            const double x = matrix[row * Columns + left];
            const double y = matrix[row * Columns + right];
            matrix[row * Columns + left]  = x * cosine + y * sine;
            matrix[row * Columns + right] = y * cosine - x * sine;
        }
    }

    // Fixed-array Golub--Reinsch SVD.  The reduction follows DGEBD2's upper
    // bidiagonal path; the implicit QR iteration is the vector-accumulating
    // form used by DGESVD/DBDSQR.  The shapes are template parameters, so
    // temporary storage and all loop bounds remain compile-time fixed.
    template <size_t Rows, size_t Columns>
    constexpr int golub_reinsch_tall(const double* source, double* u, double* singular_values, double* vt)
    {
        static_assert(Rows >= Columns, "the tall Golub-Reinsch kernel requires Rows >= Columns");

        Matrix<double, Rows, Columns> reduced{tensor::uninitialized};
        Vector<double, Columns>       off_diagonal{};
        Vector<double, Columns>       tauq{};
        Vector<double, Columns>       taup{};
        Matrix<double, Columns, Columns> v{tensor::uninitialized};
        std::copy(source, source + Rows * Columns, reduced.data());

        double* bidiagonal = reduced.data();
        double* e          = off_diagonal.data();
        double* tau_q      = tauq.data();
        double* tau_p      = taup.data();

        // DGEBD2('upper'): H(i) is stored below the diagonal and G(i) to
        // the right of the superdiagonal.  Do not overwrite either before
        // the corresponding orthogonal factor has been generated below.
        for (size_t index = 0; index < Columns; ++index)
        {
            double* left_tail = index + 1 < Rows ? bidiagonal + (index + 1) * Columns + index
                                                  : bidiagonal + index * Columns + index;
            larfg(Rows - index, bidiagonal[index * Columns + index], left_tail, Columns, tau_q[index]);
            singular_values[index] = bidiagonal[index * Columns + index];

            if (index + 1 < Columns)
            {
                if (tau_q[index] != 0.0)
                {
                    const size_t trailing = Columns - index - 1;
                    // tau_p[index + 1:] belongs to reflectors that have not
                    // been formed yet.  Reuse it as the fixed work tail and
                    // accumulate v^T A a source row at a time, so both the
                    // work vector and every matrix tail are contiguous.
                    double*       work_tail    = tau_p + index + 1;
                    const double* leading_tail = bidiagonal + index * Columns + index + 1;
                    for (size_t offset = 0; offset < trailing; ++offset)
                        work_tail[offset] = leading_tail[offset];
                    for (size_t row = index + 1; row < Rows; ++row)
                    {
                        const double  multiplier = bidiagonal[row * Columns + index];
                        const double* source_tail = bidiagonal + row * Columns + index + 1;
                        for (size_t offset = 0; offset < trailing; ++offset)
                            work_tail[offset] += multiplier * source_tail[offset];
                    }
                    for (size_t offset = 0; offset < trailing; ++offset)
                        work_tail[offset] *= tau_q[index];
                }
                if (tau_q[index] != 0.0)
                {
                    const size_t trailing = Columns - index - 1;
                    double*      work_tail = tau_p + index + 1;
                    double*      leading_tail = bidiagonal + index * Columns + index + 1;
                    for (size_t offset = 0; offset < trailing; ++offset)
                        leading_tail[offset] -= work_tail[offset];
                    for (size_t row = index + 1; row < Rows; ++row)
                    {
                        const double multiplier = bidiagonal[row * Columns + index];
                        double*      row_tail   = bidiagonal + row * Columns + index + 1;
                        for (size_t offset = 0; offset < trailing; ++offset)
                            row_tail[offset] -= multiplier * work_tail[offset];
                    }
                }

                const size_t right_count = Columns - index - 1;
                double* right_tail = index + 2 < Columns ? bidiagonal + index * Columns + index + 2
                                                          : bidiagonal + index * Columns + index + 1;
                larfg(right_count, bidiagonal[index * Columns + index + 1], right_tail, 1, tau_p[index]);
                e[index + 1] = bidiagonal[index * Columns + index + 1];

                if (tau_p[index] != 0.0)
                    for (size_t row = index + 1; row < Rows; ++row)
                    {
                        double dot = bidiagonal[row * Columns + index + 1];
                        for (size_t column = index + 2; column < Columns; ++column)
                            dot += bidiagonal[row * Columns + column] * bidiagonal[index * Columns + column];
                        dot *= tau_p[index];
                        bidiagonal[row * Columns + index + 1] -= dot;
                        for (size_t column = index + 2; column < Columns; ++column)
                            bidiagonal[row * Columns + column] -= bidiagonal[index * Columns + column] * dot;
                    }
            }
        }

        set_identity<Rows, Rows>(u);
        for (size_t index = 0; index < Columns; ++index)
        {
            if (tau_q[index] == 0.0)
                continue;
            for (size_t row = 0; row < Rows; ++row)
            {
                double dot = u[row * Rows + index];
                for (size_t reflector_row = index + 1; reflector_row < Rows; ++reflector_row)
                    dot += u[row * Rows + reflector_row] * bidiagonal[reflector_row * Columns + index];
                dot *= tau_q[index];
                u[row * Rows + index] -= dot;
                for (size_t reflector_row = index + 1; reflector_row < Rows; ++reflector_row)
                    u[row * Rows + reflector_row] -= bidiagonal[reflector_row * Columns + index] * dot;
            }
        }

        set_identity<Columns, Columns>(v.data());
        for (size_t index = 0; index + 1 < Columns; ++index)
        {
            if (tau_p[index] == 0.0)
                continue;
            const size_t offset = index + 1;
            for (size_t row = 0; row < Columns; ++row)
            {
                double dot = v[row * Columns + offset];
                for (size_t reflector_column = offset + 1; reflector_column < Columns; ++reflector_column)
                    dot += v[row * Columns + reflector_column] * bidiagonal[index * Columns + reflector_column];
                dot *= tau_p[index];
                v[row * Columns + offset] -= dot;
                for (size_t reflector_column = offset + 1; reflector_column < Columns; ++reflector_column)
                    v[row * Columns + reflector_column] -= bidiagonal[index * Columns + reflector_column] * dot;
            }
        }

        double bidiagonal_norm = 0.0;
        for (size_t index = 0; index < Columns; ++index)
            bidiagonal_norm = math::max(bidiagonal_norm, math::abs(singular_values[index]) + math::abs(e[index]));
        const double convergence_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * bidiagonal_norm;

        int info = 0;
        for (size_t remaining = Columns; remaining > 0; --remaining)
        {
            const size_t k = remaining - 1;
            bool         converged = false;
            for (size_t iteration = 0; iteration < 6 * Columns; ++iteration)
            {
                size_t l = 0;
                bool   cancel_rotation = false;
                for (size_t scan = k + 1; scan > 0; --scan)
                {
                    l = scan - 1;
                    if (math::abs(e[l]) <= convergence_tolerance)
                        break;
                    if (l == 0)
                        break;
                    if (math::abs(singular_values[l - 1]) <= convergence_tolerance)
                    {
                        cancel_rotation = true;
                        break;
                    }
                }

                if (cancel_rotation)
                {
                    double cosine = 0.0;
                    double sine   = 1.0;
                    for (size_t index = l; index <= k; ++index)
                    {
                        const double f = sine * e[index];
                        e[index]       = cosine * e[index];
                        if (math::abs(f) <= convergence_tolerance)
                            break;
                        const double g = singular_values[index];
                        const double h = math::hypot(f, g);
                        singular_values[index] = h;
                        if (h == 0.0)
                        {
                            cosine = 1.0;
                            sine   = 0.0;
                        }
                        else
                        {
                            cosine = g / h;
                            sine   = -f / h;
                        }
                        rotate_columns<Rows, Rows>(u, index - 1, index, cosine, sine);
                    }
                }

                const double z = singular_values[k];
                if (l == k)
                {
                    if (z < 0.0)
                    {
                        singular_values[k] = -z;
                        for (size_t row = 0; row < Columns; ++row)
                            v[row * Columns + k] = -v[row * Columns + k];
                    }
                    converged = true;
                    break;
                }

                const size_t km1 = k - 1;
                double x         = singular_values[l];
                const double y   = singular_values[km1];
                double g         = e[km1];
                double h         = e[k];
                double f         = 0.0;
                const double denominator = 2.0 * h * y;
                if (x != 0.0 && denominator != 0.0)
                {
                    f = ((y - z) * (y + z) + (g - h) * (g + h)) / denominator;
                    const double shift_denominator = f + signed_magnitude(math::hypot(f, 1.0), f);
                    if (shift_denominator != 0.0)
                        f = ((x - z) * (x + z) + h * (y / shift_denominator - h)) / x;
                    else
                        f = 0.0;
                }

                double cosine = 1.0;
                double sine   = 1.0;
                for (size_t index = l; index < k; ++index)
                {
                    const size_t next = index + 1;
                    g                 = e[next];
                    const double next_singular = singular_values[next];
                    h                 = sine * g;
                    g                 = cosine * g;
                    double rotation_norm = math::hypot(f, h);
                    e[index]           = rotation_norm;
                    if (rotation_norm == 0.0)
                    {
                        cosine = 1.0;
                        sine   = 0.0;
                    }
                    else
                    {
                        cosine = f / rotation_norm;
                        sine   = h / rotation_norm;
                    }
                    f = x * cosine + g * sine;
                    g = g * cosine - x * sine;
                    h = next_singular * sine;
                    const double y_rotated = next_singular * cosine;
                    rotate_columns<Columns, Columns>(v.data(), index, next, cosine, sine);

                    rotation_norm          = math::hypot(f, h);
                    singular_values[index] = rotation_norm;
                    if (rotation_norm == 0.0)
                    {
                        cosine = 1.0;
                        sine   = 0.0;
                    }
                    else
                    {
                        cosine = f / rotation_norm;
                        sine   = h / rotation_norm;
                    }
                    f = cosine * g + sine * y_rotated;
                    x = cosine * y_rotated - sine * g;
                    rotate_columns<Rows, Rows>(u, index, next, cosine, sine);
                }
                e[l]                = 0.0;
                e[k]                = f;
                singular_values[k]  = x;
            }
            if (!converged)
            {
                info = static_cast<int>(k + 1);
                break;
            }
        }

        // DGESVD returns non-negative singular values in decreasing order.
        // The corresponding column exchanges preserve A = U*S*V^T.
        for (size_t index = 0; index < Columns; ++index)
        {
            size_t largest = index;
            for (size_t candidate = index + 1; candidate < Columns; ++candidate)
                if (singular_values[candidate] > singular_values[largest])
                    largest = candidate;
            if (largest == index)
                continue;
            std::swap(singular_values[index], singular_values[largest]);
            for (size_t row = 0; row < Rows; ++row)
                std::swap(u[row * Rows + index], u[row * Rows + largest]);
            for (size_t row = 0; row < Columns; ++row)
                std::swap(v[row * Columns + index], v[row * Columns + largest]);
        }

        for (size_t row = 0; row < Columns; ++row)
            for (size_t column = 0; column < Columns; ++column)
                vt[row * Columns + column] = v[column * Columns + row];
        return info;
    }

    template <size_t Rows, size_t Columns>
    constexpr int golub_reinsch_factorize(const double* source, double* u, double* singular_values, double* vt)
    {
        if constexpr (Rows >= Columns)
        {
            return golub_reinsch_tall<Rows, Columns>(source, u, singular_values, vt);
        }
        else
        {
            Matrix<double, Columns, Rows> transposed{tensor::uninitialized};
            Matrix<double, Columns, Columns> transposed_u{tensor::uninitialized};
            Matrix<double, Rows, Rows> transposed_vt{tensor::uninitialized};
            Vector<double, Rows> transposed_s{tensor::uninitialized};
            for (size_t row = 0; row < Rows; ++row)
                for (size_t column = 0; column < Columns; ++column)
                    transposed[column * Rows + row] = source[row * Columns + column];

            const int info = golub_reinsch_tall<Columns, Rows>(
                transposed.data(), transposed_u.data(), transposed_s.data(), transposed_vt.data());
            for (size_t row = 0; row < Rows; ++row)
                for (size_t column = 0; column < Rows; ++column)
                    u[row * Rows + column] = transposed_vt[column * Rows + row];
            for (size_t index = 0; index < Rows; ++index)
                singular_values[index] = transposed_s[index];
            for (size_t row = 0; row < Columns; ++row)
                for (size_t column = 0; column < Columns; ++column)
                    vt[row * Columns + column] = transposed_u[column * Columns + row];
            return info;
        }
    }

    struct BunchKaufman
    {
        template <size_t N1, size_t N2>
        struct Context;

    private:
        static int  lapack_factorize_inplace(int, double*, int, int*);
        static void lapack_substitute_inplace(int, int, const double*, int, const int*, double*, int);
    };

    struct Cholesky
    {
        template <size_t N1, size_t N2>
        struct Context;

    private:
        static int  lapack_factorize_inplace(int, double*, int);
        static void lapack_substitute_inplace(int, int, const double*, int, double*, int);
    };

    struct Doolittle
    {
        template <size_t N1, size_t N2>
        struct Context;

    private:
        static int  lapack_factorize_inplace(int, int, double*, int, int*);
        static void lapack_substitute_inplace(int, int, const double*, int, const int*, double*, int);
    };

    struct GolubReinsch
    {
        template <size_t N1, size_t N2>
        struct Context;

    private:
        static int lapack_factorize_inplace(int, int, const double*, double*, double*, double*, double*);
        static void
        lapack_substitute_inplace(int, int, const double*, const double*, const double*, double*, int, double*);
    };

    struct Householder
    {
        template <size_t N1, size_t N2>
        struct Context;

    private:
        static int  lapack_factorize_inplace(int, int, double*, int, double*);
        static void lapack_substitute_inplace(int, int, int, const double*, int, const double*, double*, int);
    };

    struct Thomas
    {
        template <size_t Bandwidth, size_t N>
        struct Context;

    private:
        static int lapack_factorize_inplace(int, int, int, double*, int, int*);
        static void lapack_substitute_inplace(int, int, int, int, const double*, int, const int*, double*, int);
    };

    // The VEQ artifact dimensions are template arguments.  These predicates
    // are therefore folded at build time: a small artifact contains only the
    // fixed-storage kernel, while a larger one calls the blocked LAPACKE path
    // at runtime.  Each handoff remains a policy-level tuning constant rather
    // than a dynamic dimension branch.
    inline constexpr size_t doolittle_lapack_min_order     = 128;
    inline constexpr size_t bunch_kaufman_lapack_min_order = 64;
    inline constexpr size_t cholesky_lapack_min_order      = 128;
    inline constexpr size_t householder_lapack_min_order   = 128;
    inline constexpr size_t golub_reinsch_lapack_min_order = 64;
    // Thomas remains constexpr for every realistic VEQ grid.  The fixed 8096
    // handoff preserves a standard LAPACKE route for larger future band systems
    // without creating a second small-N implementation policy.
    inline constexpr size_t thomas_lapack_min_order = 8096;

    template <typename Policy, size_t N1, size_t N2>
    inline constexpr bool uses_runtime_lapack = runtime_lapack_enabled && []
    {
        if constexpr (std::is_same_v<Policy, Doolittle>)
            return N1 >= doolittle_lapack_min_order;
        else if constexpr (std::is_same_v<Policy, BunchKaufman>)
            return N1 >= bunch_kaufman_lapack_min_order;
        else if constexpr (std::is_same_v<Policy, Cholesky>)
            return N1 >= cholesky_lapack_min_order;
        else if constexpr (std::is_same_v<Policy, Householder>)
            return N1 >= householder_lapack_min_order;
        else if constexpr (std::is_same_v<Policy, GolubReinsch>)
            return std::max(N1, N2) >= golub_reinsch_lapack_min_order;
        else if constexpr (std::is_same_v<Policy, Thomas>)
            return N2 >= thomas_lapack_min_order;
        else
            return false;
    }();

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

            if constexpr (uses_runtime_lapack<BunchKaufman, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    info = lapack_factorize_inplace(N1, L, N1, ipiv);
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
                            // The lower-triangular Schur complement has one
                            // independent update per stored entry.  Keeping a
                            // destination row contiguous avoids the
                            // column-stride stores of the literal DSYTF2
                            // loop.  Delay scaling L(:, k) until every
                            // update has consumed its original value.
                            for (size_t row = k + 1; row < N1; ++row)
                            {
                                const double row_k = L[row * N1 + k];
                                double*      row_tail = L + row * N1 + k + 1;
                                for (size_t column = k + 1; column <= row; ++column)
                                    row_tail[column - k - 1] -= row_k * L[column * N1 + k] * d11;
                            }
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

                        // A row reaches its diagonal only after the W terms
                        // for all prior columns have been stored below.  This
                        // is algebraically the same lower-triangle traversal
                        // as DSYTF2, but makes each destination tail
                        // contiguous in row-major storage.
                        for (size_t row = k + 2; row < N1; ++row)
                        {
                            const double row_k   = L[row * N1 + k];
                            const double row_kp1 = L[row * N1 + (k + 1)];
                            const double wk      = d21 * (d11 * row_k - row_kp1);
                            const double wkp1    = d21 * (d22 * row_kp1 - row_k);
                            double*      row_tail = L + row * N1 + k + 2;
                            for (size_t column = k + 2; column < row; ++column)
                                row_tail[column - k - 2] -=
                                    row_k * L[column * N1 + k] + row_kp1 * L[column * N1 + (k + 1)];
                            L[row * N1 + row] -= row_k * wk + row_kp1 * wkp1;
                            L[row * N1 + k]       = wk;
                            L[row * N1 + (k + 1)] = wkp1;
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

            if constexpr (uses_runtime_lapack<BunchKaufman, N1, N2>)
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
                    {
                        const double  multiplier_k   = L[row * N1 + k];
                        const double  multiplier_kp1 = L[row * N1 + (k + 1)];
                        const double* rhs_k          = x + k * P;
                        const double* rhs_kp1        = x + (k + 1) * P;
                        double*       rhs_row        = x + row * P;
                        for (size_t rhs = 0; rhs < P; ++rhs)
                        {
                            rhs_row[rhs] -= multiplier_k * rhs_k[rhs];
                            rhs_row[rhs] -= multiplier_kp1 * rhs_kp1[rhs];
                        }
                    }

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
                    double* rhs_k = x + k * P;
                    for (size_t row = k + 1; row < N1; ++row)
                    {
                        const double  multiplier = L[row * N1 + k];
                        const double* rhs_row    = x + row * P;
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            rhs_k[rhs] -= multiplier * rhs_row[rhs];
                    }
                    const size_t kp = static_cast<size_t>(ipiv[k] - 1);
                    swap_rhs_rows(x, k, kp, P);
                    remaining = k;
                }
                else
                {
                    const size_t km1 = k - 1;
                    double* rhs_k   = x + k * P;
                    double* rhs_km1 = x + km1 * P;
                    for (size_t row = k + 1; row < N1; ++row)
                    {
                        const double  multiplier_k   = L[row * N1 + k];
                        const double  multiplier_km1 = L[row * N1 + km1];
                        const double* rhs_row        = x + row * P;
                        for (size_t rhs = 0; rhs < P; ++rhs)
                        {
                            rhs_k[rhs] -= multiplier_k * rhs_row[rhs];
                            rhs_km1[rhs] -= multiplier_km1 * rhs_row[rhs];
                        }
                    }
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

            if constexpr (uses_runtime_lapack<Cholesky, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    info = lapack_factorize_inplace(N1, L, N1);
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

            if constexpr (uses_runtime_lapack<Cholesky, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, L, N1, x, P);
                    return;
                }
            }

            for (size_t i = 0; i < N1; ++i)
            {
                const double* factor_row = L + i * N1;
                double*       rhs_row    = x + i * P;
                for (size_t k = 0; k < i; ++k)
                {
                    const double  multiplier = factor_row[k];
                    const double* solved_row = x + k * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * solved_row[j];
                }
                for (size_t j = 0; j < P; ++j)
                    rhs_row[j] /= factor_row[i];
            }

            for (size_t ii = N1; ii > 0; --ii)
            {
                const size_t i = ii - 1;
                double* rhs_row = x + i * P;
                for (size_t k = i + 1; k < N1; ++k)
                {
                    const double  multiplier = L[k * N1 + i];
                    const double* solved_row = x + k * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * solved_row[j];
                }
                for (size_t j = 0; j < P; ++j)
                    rhs_row[j] /= L[i * N1 + i];
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

            if constexpr (uses_runtime_lapack<Doolittle, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    info = lapack_factorize_inplace(N1, N1, LU, N1, ipiv);
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
                }
                else if (info == 0)
                {
                    info = static_cast<int>(k + 1);
                }

                if (k + 1 < N1)
                {
                    // Scaling the current column and applying its rank-one
                    // update have no cross-row dependency.  Fuse them so a
                    // row reaches its contiguous trailing tail only once;
                    // safe-min scaling remains identical to DGETF2.
                    const size_t  trailing   = N1 - k - 1;
                    const double* pivot_tail = LU + k * N1 + k + 1;
                    const double  pivot      = LU[k * N1 + k];
                    if (pivot != 0.0 && math::abs(pivot) >= lapack_safe_min)
                    {
                        const double reciprocal_pivot = 1.0 / pivot;
                        for (size_t row = k + 1; row < N1; ++row)
                        {
                            const double multiplier = LU[row * N1 + k] * reciprocal_pivot;
                            LU[row * N1 + k]         = multiplier;
                            double* row_tail = LU + row * N1 + k + 1;
                            for (size_t offset = 0; offset < trailing; ++offset)
                                row_tail[offset] -= multiplier * pivot_tail[offset];
                        }
                    }
                    else if (pivot != 0.0)
                    {
                        for (size_t row = k + 1; row < N1; ++row)
                        {
                            const double multiplier = LU[row * N1 + k] / pivot;
                            LU[row * N1 + k]         = multiplier;
                            double* row_tail = LU + row * N1 + k + 1;
                            for (size_t offset = 0; offset < trailing; ++offset)
                                row_tail[offset] -= multiplier * pivot_tail[offset];
                        }
                    }
                    else
                    {
                        for (size_t row = k + 1; row < N1; ++row)
                        {
                            const double multiplier = LU[row * N1 + k];
                            double*      row_tail   = LU + row * N1 + k + 1;
                            for (size_t offset = 0; offset < trailing; ++offset)
                                row_tail[offset] -= multiplier * pivot_tail[offset];
                        }
                    }
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* LU   = LU_mat.data();
            const int*    ipiv = pivot_vec.data();

            if constexpr (uses_runtime_lapack<Doolittle, N1, N2>)
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
            {
                const double* factor_row = LU + i * N1;
                double*       rhs_row    = x + i * P;
                for (size_t k = 0; k < i; ++k)
                {
                    const double  multiplier = factor_row[k];
                    const double* solved_row = x + k * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * solved_row[j];
                }
            }

            for (size_t i = N1; i-- > 0;)
            {
                const double* factor_row = LU + i * N1;
                double*       rhs_row    = x + i * P;
                for (size_t k = i + 1; k < N1; ++k)
                {
                    const double  multiplier = factor_row[k];
                    const double* solved_row = x + k * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * solved_row[j];
                }
                for (size_t j = 0; j < P; ++j)
                    rhs_row[j] /= factor_row[i];
            }
        }
    };

    template <size_t N1, size_t N2>
    struct GolubReinsch::Context
    {
        Matrix<double, N1, N1>           U_mat{tensor::uninitialized};
        Matrix<double, N2, N2>           Vt_mat{tensor::uninitialized};
        Vector<double, std::min(N1, N2)> S_vec{tensor::uninitialized};
        int                               info{0};

        constexpr void factorize_from(const double* A)
        {
            double* U  = U_mat.data();
            double* Vt = Vt_mat.data();
            double* S  = S_vec.data();

            if constexpr (uses_runtime_lapack<GolubReinsch, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    Matrix<double, N1, N2> work{tensor::uninitialized};
                    info = lapack_factorize_inplace(N1, N2, A, work.data(), U, S, Vt);
                    return;
                }
            }

            info       = golub_reinsch_factorize<N1, N2>(A, U, S, Vt);
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* U  = U_mat.data();
            const double* Vt = Vt_mat.data();
            const double* S  = S_vec.data();
            constexpr size_t rank = std::min(N1, N2);

            if constexpr (uses_runtime_lapack<GolubReinsch, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    Matrix<double, std::max(N1, N2), P> work{tensor::uninitialized};
                    lapack_substitute_inplace(N1, N2, U, S, Vt, x, P, work.data());
                    return;
                }
            }

            Matrix<double, N1, P> projected{};
            Matrix<double, N2, P> spectral{};
            Matrix<double, N2, P> result{};

            // U is stored by rows, whereas the pseudoinverse needs U^T x.
            // Accumulating one source row at a time makes every U read
            // contiguous while retaining the original source-row summation
            // order for each projected component.
            for (size_t source_row = 0; source_row < N1; ++source_row)
            {
                const double* u_row   = U + source_row * N1;
                const double* rhs_row = x + source_row * P;
                for (size_t projected_row = 0; projected_row < N1; ++projected_row)
                {
                    const double multiplier = u_row[projected_row];
                    double*      output_row = projected.data() + projected_row * P;
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        output_row[rhs] += multiplier * rhs_row[rhs];
                }
            }

            // Preserve the legacy VEQ pseudoinverse cutoff while removing
            // the CBLAS/DGESDD implementation that previously imposed it.
            for (size_t index = 0; index < rank; ++index)
                if (S[index] > 1.0e-12)
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        spectral[index * P + rhs] = projected[index * P + rhs] / S[index];

            // Vt(source, :) is one contiguous row in storage and represents
            // column `source` of V.  This row-wise accumulation computes
            // V * spectral without striding through Vt's columns.
            for (size_t source_row = 0; source_row < N2; ++source_row)
            {
                const double* vt_row       = Vt + source_row * N2;
                const double* spectral_row = spectral.data() + source_row * P;
                for (size_t result_row = 0; result_row < N2; ++result_row)
                {
                    const double multiplier = vt_row[result_row];
                    double*      output_row = result.data() + result_row * P;
                    for (size_t rhs = 0; rhs < P; ++rhs)
                        output_row[rhs] += multiplier * spectral_row[rhs];
                }
            }
            std::copy(result.data(), result.data() + result.count, x);
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

            if constexpr (uses_runtime_lapack<Householder, N1, N2>)
            {
                if (!std::is_constant_evaluated())
                {
                    info = lapack_factorize_inplace(N1, N2, QR, N2, tau);
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
                    // Form v^T A first, then visit every target row once.
                    // The reflector vector remains a column (and therefore
                    // has an unavoidable stride), but the trailing block is
                    // read and written through contiguous row tails.
                    const size_t trailing = N2 - k - 1;
                    double*      work_tail = tau + k + 1;
                    double*      leading_tail = QR + k * N2 + k + 1;
                    for (size_t offset = 0; offset < trailing; ++offset)
                        work_tail[offset] = leading_tail[offset];
                    for (size_t row = k + 1; row < N1; ++row)
                    {
                        const double  multiplier = QR[row * N2 + k];
                        const double* source_tail = QR + row * N2 + k + 1;
                        for (size_t offset = 0; offset < trailing; ++offset)
                            work_tail[offset] += multiplier * source_tail[offset];
                    }
                    for (size_t offset = 0; offset < trailing; ++offset)
                    {
                        work_tail[offset] *= tau[k];
                        leading_tail[offset] -= work_tail[offset];
                    }
                    for (size_t row = k + 1; row < N1; ++row)
                    {
                        const double multiplier = QR[row * N2 + k];
                        double*      row_tail   = QR + row * N2 + k + 1;
                        for (size_t offset = 0; offset < trailing; ++offset)
                            row_tail[offset] -= multiplier * work_tail[offset];
                    }
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* QR  = QR_mat.data();
            const double* tau = tau_vec.data();

            if constexpr (uses_runtime_lapack<Householder, N1, N2>)
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

                Vector<double, P> reflector_work{tensor::uninitialized};
                for (size_t j = 0; j < P; ++j)
                    reflector_work[j] = x[k * P + j];
                for (size_t i = k + 1; i < N1; ++i)
                {
                    const double  multiplier = QR[i * N2 + k];
                    const double* rhs_row    = x + i * P;
                    for (size_t j = 0; j < P; ++j)
                        reflector_work[j] += multiplier * rhs_row[j];
                }
                for (size_t j = 0; j < P; ++j)
                {
                    reflector_work[j] *= tau[k];
                    x[k * P + j] -= reflector_work[j];
                }
                for (size_t i = k + 1; i < N1; ++i)
                {
                    const double multiplier = QR[i * N2 + k];
                    double*      rhs_row    = x + i * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * reflector_work[j];
                }
            }

            for (size_t i = N2; i > 0; --i)
            {
                const double* factor_row = QR + (i - 1) * N2;
                double*       rhs_row    = x + (i - 1) * P;
                for (size_t column = i; column < N2; ++column)
                {
                    const double  multiplier = factor_row[column];
                    const double* solved_row = x + column * P;
                    for (size_t j = 0; j < P; ++j)
                        rhs_row[j] -= multiplier * solved_row[j];
                }
                for (size_t j = 0; j < P; ++j)
                    rhs_row[j] /= factor_row[i - 1];
            }
        }
    };

    template <size_t Bandwidth, size_t N>
    struct Thomas::Context
    {
        static_assert(Bandwidth % 2 == 1, "Thomas band width must be odd");
        static_assert(N >= Bandwidth, "Thomas matrix must have at least Bandwidth rows");

        static constexpr size_t radius = Bandwidth / 2;
        static constexpr size_t lapack_ldab = 3 * radius + 1;

        Matrix<double, Bandwidth, N> LU_mat{tensor::uninitialized};
        // Physical column-major storage for DGBTRF: Matrix<N, ldab> is laid
        // out as ab[band_row + column * ldab].  It is unused by every current
        // VEQ grid, where the constexpr Thomas path remains selected.
        Matrix<double, N, lapack_ldab> lapack_ab{tensor::uninitialized};
        Vector<int, N>                 lapack_ipiv{tensor::uninitialized};
        int                            info = 0;

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
            if constexpr (uses_runtime_lapack<Thomas, Bandwidth, N>)
            {
                if (!std::is_constant_evaluated())
                {
                    std::fill(lapack_ab.data(), lapack_ab.data() + lapack_ab.count, 0.0);
                    for (size_t column = 0; column < N; ++column)
                    {
                        const size_t row_start = column > radius ? column - radius : 0;
                        const size_t row_stop  = std::min(N, column + radius + 1);
                        for (size_t row = row_start; row < row_stop; ++row)
                            lapack_ab[column * lapack_ldab + 2 * radius + row - column] =
                                matrix[band_index(row, column) * N + column];
                    }
                    info = lapack_factorize_inplace(static_cast<int>(N),
                                                     static_cast<int>(radius),
                                                     static_cast<int>(radius),
                                                     lapack_ab.data(),
                                                     static_cast<int>(lapack_ldab),
                                                     lapack_ipiv.data());
                    assert(info == 0);
                    return;
                }
            }

            info = 0;
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
            if constexpr (uses_runtime_lapack<Thomas, Bandwidth, N>)
            {
                if (!std::is_constant_evaluated())
                {
                    Matrix<double, P, N> rhs_col_major{tensor::uninitialized};
                    for (size_t row = 0; row < N; ++row)
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            rhs_col_major[rhs * N + row] = x[row * P + rhs];

                    lapack_substitute_inplace(static_cast<int>(N),
                                               static_cast<int>(radius),
                                               static_cast<int>(radius),
                                               static_cast<int>(P),
                                               lapack_ab.data(),
                                               static_cast<int>(lapack_ldab),
                                               lapack_ipiv.data(),
                                               rhs_col_major.data(),
                                               static_cast<int>(N));

                    for (size_t row = 0; row < N; ++row)
                        for (size_t rhs = 0; rhs < P; ++rhs)
                            x[row * P + rhs] = rhs_col_major[rhs * N + row];
                    return;
                }
            }

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
