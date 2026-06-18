#pragma once

#include "math.h"
#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace linalg::detail
{
    using std::size_t;
    using math::sqrt;
    using tensor::Matrix;
    using tensor::Vector;

    template <typename Policy, size_t N1, size_t N2>
    using Context = typename Policy::template Context<N1, N2>;

    inline constexpr size_t doolittle_lapack_min_size     = 128 * 128;
    inline constexpr size_t cholesky_lapack_min_size      = 16 * 16;
    inline constexpr size_t bunch_kaufman_lapack_min_size = 96 * 96;
    inline constexpr size_t householder_lapack_min_size   = 48 * 48;

    inline constexpr double linear_tolerance = 1.0e-10;

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

        static constexpr double alpha = 0.6403882032;

        Matrix<double, N1, N1> LDLT_mat{tensor::uninitialized};
        Vector<int, N1>        pivot_vec{tensor::uninitialized};

        constexpr void factorize_from(const double* A)
        {
            double* L    = LDLT_mat.data();
            int*    ipiv = pivot_vec.data();
            std::copy(A, A + N1 * N1, L);

            if constexpr (N1 * N1 >= bunch_kaufman_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, L, N1, ipiv);
                    return;
                }
            }

            for (size_t k = 0; k < N1;)
            {
                size_t p            = k;
                double max_off_diag = 0.0;
                for (size_t i = k + 1; i < N1; ++i)
                {
                    if (std::abs(L[i * N1 + k]) > max_off_diag)
                    {
                        max_off_diag = std::abs(L[i * N1 + k]);
                        p            = i;
                    }
                }

                if (std::abs(L[k * N1 + k]) >= alpha * max_off_diag)
                {
                    ipiv[k] = static_cast<int>(k);

                    double dk = L[k * N1 + k];
                    if (std::abs(dk) > linear_tolerance)
                        for (size_t i = k + 1; i < N1; ++i)
                            L[i * N1 + k] /= dk;

                    for (size_t i = k + 1; i < N1; ++i)
                    {
                        double lik = L[i * N1 + k];
                        for (size_t j = k + 1; j <= i; ++j)
                            L[i * N1 + j] -= lik * L[j * N1 + k] * dk;
                    }

                    k++;
                }
                else
                {
                    if (p != k + 1)
                    {
                        for (size_t i = 0; i < N1; ++i)
                            std::swap(L[p * N1 + i], L[(k + 1) * N1 + i]);

                        for (size_t i = 0; i < N1; ++i)
                            std::swap(L[i * N1 + p], L[i * N1 + (k + 1)]);
                    }

                    ipiv[k]     = static_cast<int>(p);
                    ipiv[k + 1] = static_cast<int>(p);

                    double d11 = L[k * N1 + k];
                    double d21 = L[(k + 1) * N1 + k];
                    double d22 = L[(k + 1) * N1 + (k + 1)];

                    double det = d11 * d22 - d21 * d21;
                    assert(std::abs(det) >= linear_tolerance);
                    double inv_d11 = d22 / det;
                    double inv_d21 = -d21 / det;
                    double inv_d22 = d11 / det;

                    for (size_t i = k + 2; i < N1; ++i)
                    {
                        double lik        = L[i * N1 + k];
                        double lik1       = L[i * N1 + k + 1];
                        L[i * N1 + k]     = (lik * inv_d11 + lik1 * inv_d21);
                        L[i * N1 + k + 1] = (lik * inv_d21 + lik1 * inv_d22);
                    }

                    for (size_t i = k + 2; i < N1; ++i)
                        for (size_t j = k + 2; j <= i; ++j)
                            L[i * N1 + j] -=
                                (L[i * N1 + k] * L[j * N1 + k] * d11 +
                                 (L[i * N1 + k] * L[j * N1 + k + 1] + L[i * N1 + k + 1] * L[j * N1 + k]) * d21 +
                                 L[i * N1 + k + 1] * L[j * N1 + k + 1] * d22);

                    k += 2;
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* L    = LDLT_mat.data();
            const int*    ipiv = pivot_vec.data();

            if constexpr (N1 * N1 >= bunch_kaufman_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, L, N1, ipiv, x, P);
                    return;
                }
            }

            for (size_t k = 0; k < N1;)
            {
                if (ipiv[k] == static_cast<int>(k))
                {
                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = k + 1; i < N1; ++i)
                            x[i * P + j] -= L[i * N1 + k] * x[k * P + j];
                    k++;
                }
                else
                {
                    size_t p = static_cast<size_t>(ipiv[k]);
                    if (p != k + 1)
                        for (size_t j = 0; j < P; ++j)
                            std::swap(x[(k + 1) * P + j], x[p * P + j]);

                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = k + 2; i < N1; ++i)
                            x[i * P + j] -= L[i * N1 + k] * x[k * P + j] + L[i * N1 + k + 1] * x[(k + 1) * P + j];
                    k += 2;
                }
            }

            for (size_t k = N1; k > 0;)
            {
                --k;
                if (ipiv[k] == static_cast<int>(k))
                {
                    double dk = L[k * N1 + k];
                    for (size_t j = 0; j < P; ++j)
                        x[k * P + j] /= dk;
                }
                else
                {
                    assert(k > 0);
                    const size_t km1 = k - 1;
                    double       d11 = L[km1 * N1 + km1];
                    double       d21 = L[k * N1 + km1];
                    double       d22 = L[k * N1 + k];
                    double       det = d11 * d22 - d21 * d21;
                    assert(std::abs(det) >= linear_tolerance);
                    for (size_t j = 0; j < P; ++j)
                    {
                        double y1      = x[km1 * P + j];
                        double y2      = x[k * P + j];
                        x[km1 * P + j] = (y1 * d22 - y2 * d21) / det;
                        x[k * P + j]   = (y2 * d11 - y1 * d21) / det;
                    }
                    k = km1;
                }
            }

            for (size_t k = N1; k > 0;)
            {
                --k;
                if (ipiv[k] == static_cast<int>(k))
                {
                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = k + 1; i < N1; ++i)
                            x[k * P + j] -= L[i * N1 + k] * x[i * P + j];
                }
                else
                {
                    assert(k > 0);
                    const size_t km1 = k - 1;
                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = k + 1; i < N1; ++i)
                        {
                            x[k * P + j] -= L[i * N1 + k] * x[i * P + j];
                            x[km1 * P + j] -= L[i * N1 + km1] * x[i * P + j];
                        }

                    size_t p = static_cast<size_t>(ipiv[k]);
                    if (p != k)
                        for (size_t j = 0; j < P; ++j)
                            std::swap(x[k * P + j], x[p * P + j]);
                    k = km1;
                }
            }
        }
    };

    template <size_t N1, size_t N2>
    struct Cholesky::Context
    {
        static_assert(N1 == N2, "cholesky only supports square matrices");

        Matrix<double, N1, N1> LLT_mat{tensor::uninitialized};

        constexpr void factorize_from(const double* A)
        {
            double* L = LLT_mat.data();
            std::copy(A, A + N1 * N1, L);

            if constexpr (N1 * N1 >= cholesky_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, L, N1);
                    return;
                }
            }

            for (size_t i = 0; i < N1; ++i)
                for (size_t j = i + 1; j < N1; ++j)
                    assert(std::abs(L[i * N1 + j] - L[j * N1 + i]) <= linear_tolerance);

            for (size_t i = 0; i < N1; ++i)
            {
                for (size_t j = 0; j < i; ++j)
                {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += L[i * N1 + k] * L[j * N1 + k];

                    L[i * N1 + j] = (L[i * N1 + j] - sum) / L[j * N1 + j];
                }

                double sum = 0.0;
                for (size_t k = 0; k < i; ++k)
                    sum += L[i * N1 + k] * L[i * N1 + k];

                double diag_val = L[i * N1 + i] - sum;
                assert(diag_val > linear_tolerance);

                L[i * N1 + i] = sqrt(diag_val);
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* L = LLT_mat.data();

            if constexpr (N1 * N1 >= cholesky_lapack_min_size)
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

        constexpr void factorize_from(const double* A)
        {
            double* LU   = LU_mat.data();
            int*    ipiv = pivot_vec.data();
            std::copy(A, A + N1 * N1, LU);

            if constexpr (N1 * N1 >= doolittle_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, N1, LU, N1, ipiv);
                    return;
                }
            }

            for (size_t k = 0; k < N1 - 1; ++k)
            {
                size_t pivot_row = k;
                double max_norm  = std::abs(LU[k * N1 + k]);

                for (size_t i = k + 1; i < N1; ++i)
                {
                    double curr_norm = std::abs(LU[i * N1 + k]);
                    if (curr_norm > max_norm)
                    {
                        max_norm  = curr_norm;
                        pivot_row = i;
                    }
                }

                ipiv[k] = static_cast<int>(pivot_row);
                if (pivot_row != k)
                    std::swap_ranges(LU + k * N1, LU + (k + 1) * N1, LU + pivot_row * N1);

                for (size_t i = k + 1; i < N1; ++i)
                {
                    assert(std::abs(LU[k * N1 + k]) >= linear_tolerance);

                    double multiplier = LU[i * N1 + k] / LU[k * N1 + k];
                    LU[i * N1 + k]    = multiplier;
                    for (size_t j = k + 1; j < N1; ++j)
                        LU[i * N1 + j] -= multiplier * LU[k * N1 + j];
                }
            }

            ipiv[N1 - 1] = N1 - 1;
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* LU   = LU_mat.data();
            const int*    ipiv = pivot_vec.data();

            if constexpr (N1 * N1 >= doolittle_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, P, LU, N1, ipiv, x, P);
                    return;
                }
            }

            for (size_t k = 0; k < N1 - 1; ++k)
            {
                int pivot_row = ipiv[k];
                if (pivot_row != static_cast<int>(k))
                    std::swap_ranges(x + k * P, x + (k + 1) * P, x + static_cast<size_t>(pivot_row) * P);
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

        constexpr void factorize_from(const double* A)
        {
            double* QR  = QR_mat.data();
            double* tau = tau_vec.data();
            std::copy(A, A + N1 * N2, QR);

            if constexpr (N1 * N2 >= householder_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_factorize_inplace(N1, N2, QR, N2, tau);
                    return;
                }
            }

            for (size_t k = 0; k < N2; ++k)
            {
                double norm2 = 0.0;
                for (size_t i = k; i < N1; ++i)
                    norm2 += QR[i * N2 + k] * QR[i * N2 + k];

                double norm = sqrt(norm2);
                if (norm < linear_tolerance)
                {
                    assert(norm >= linear_tolerance);
                    tau[k] = 0.0;
                    continue;
                }

                double akk  = QR[k * N2 + k];
                double sign = (akk >= 0.0) ? 1.0 : -1.0;
                double u1   = akk + sign * norm;

                tau[k] = u1 * u1 / (norm2 + std::abs(akk * norm));

                double inv_u1 = 1.0 / u1;
                for (size_t i = k + 1; i < N1; ++i)
                    QR[i * N2 + k] *= inv_u1;

                QR[k * N2 + k] = -sign * norm;

                for (size_t j = k + 1; j < N2; ++j)
                {
                    double vTA_j = QR[k * N2 + j];
                    for (size_t i = k + 1; i < N1; ++i)
                        vTA_j += QR[i * N2 + k] * QR[i * N2 + j];

                    QR[k * N2 + j] -= tau[k] * vTA_j;
                    for (size_t i = k + 1; i < N1; ++i)
                        QR[i * N2 + j] -= tau[k] * QR[i * N2 + k] * vTA_j;
                }
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* QR  = QR_mat.data();
            const double* tau = tau_vec.data();

            if constexpr (N1 * N2 >= householder_lapack_min_size)
            {
                if (!std::is_constant_evaluated())
                {
                    lapack_substitute_inplace(N1, N2, P, QR, N2, tau, x, P);
                    return;
                }
            }

            for (size_t k = 0; k < N2; ++k)
            {
                if (std::abs(tau[k]) < linear_tolerance)
                    continue;

                for (size_t j = 0; j < P; ++j)
                {
                    double vTx_j = x[k * P + j];
                    for (size_t i = k + 1; i < N1; ++i)
                        vTx_j += QR[i * N2 + k] * x[i * P + j];

                    x[k * P + j] -= tau[k] * vTx_j;
                    for (size_t i = k + 1; i < N1; ++i)
                        x[i * P + j] -= tau[k] * QR[i * N2 + k] * vTx_j;
                }
            }

            for (size_t j = 0; j < P; ++j)
            {
                for (size_t i = N2; i > 0; --i)
                {
                    double sum = x[(i - 1) * P + j];
                    for (size_t k = i; k < N2; ++k)
                        sum -= QR[(i - 1) * N2 + k] * x[k * P + j];

                    if (std::abs(QR[(i - 1) * N2 + (i - 1)]) < linear_tolerance)
                        x[(i - 1) * P + j] = 0.0;
                    else
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
