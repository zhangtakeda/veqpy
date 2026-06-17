#pragma once

#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace linalg
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;

    template <typename Policy, size_t N1, size_t N2>
    using Context = typename Policy::template Context<N1, N2>;

    inline constexpr size_t doolittle_lapack_min_size     = 128 * 128;
    inline constexpr size_t cholesky_lapack_min_size      = 16 * 16;
    inline constexpr size_t bunch_kaufman_lapack_min_size = 96 * 96;
    inline constexpr size_t householder_lapack_min_size   = 48 * 48;

    inline constexpr double linear_tolerance = 1.0e-10;

    struct BunchKaufman;
    struct Cholesky;
    struct Doolittle;
    struct GolubReinsch;
    struct Householder;

    template <typename Policy = Doolittle, size_t N1, size_t N2>
    constexpr Context<Policy, N1, N2>& factorize_into(Context<Policy, N1, N2>&              context,
                                                      const tensor::Matrix<double, N1, N2>& matrix)
    {
        context.factorize_from(matrix.data());
        return context;
    }

    template <typename Policy = Doolittle, size_t N1, size_t N2>
    constexpr Context<Policy, N1, N2> factorize(const tensor::Matrix<double, N1, N2>& matrix)
    {
        Context<Policy, N1, N2> context{};
        context.factorize_from(matrix.data());
        return context;
    }

    template <typename Policy = Doolittle, size_t N1, size_t N2, size_t P>
    constexpr void solve_inplace(const tensor::Matrix<double, N1, N2>& matrix, tensor::Matrix<double, N1, P>& rhs)
    {
        const auto context = factorize<Policy>(matrix);
        context.template substitute_inplace<P>(rhs.data());
    }

    template <typename Policy = Doolittle, size_t N1, size_t N2, size_t P>
    constexpr void solve_into(tensor::Matrix<double, N2, P>&        solution,
                              const tensor::Matrix<double, N1, N2>& matrix,
                              const tensor::Matrix<double, N1, P>&  rhs)
    {
        if constexpr (N1 > N2)
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
    constexpr auto solve(const tensor::Matrix<double, N1, N2>& matrix, const tensor::Matrix<double, N1, P>& rhs)
    {
        tensor::Matrix<double, N2, P> solution{tensor::uninitialized};
        solve_into<Policy>(solution, matrix, rhs);
        return solution;
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
                lapack_factorize_inplace(N1, L, N1, ipiv);
                return;
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
                lapack_substitute_inplace(N1, P, L, N1, ipiv, x, P);
                return;
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

            for (int k = N1 - 1; k >= 0;)
            {
                if (ipiv[k] == k)
                {
                    double dk = L[k * N1 + k];
                    for (size_t j = 0; j < P; ++j)
                        x[k * P + j] /= dk;
                    k--;
                }
                else
                {
                    double d11 = L[(k - 1) * N1 + (k - 1)];
                    double d21 = L[k * N1 + (k - 1)];
                    double d22 = L[k * N1 + k];
                    double det = d11 * d22 - d21 * d21;
                    assert(std::abs(det) >= linear_tolerance);
                    for (size_t j = 0; j < P; ++j)
                    {
                        double y1          = x[(k - 1) * P + j];
                        double y2          = x[k * P + j];
                        x[(k - 1) * P + j] = (y1 * d22 - y2 * d21) / det;
                        x[k * P + j]       = (y2 * d11 - y1 * d21) / det;
                    }
                    k -= 2;
                }
            }

            for (int k = N1 - 1; k >= 0;)
            {
                if (ipiv[k] == k)
                {
                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = static_cast<size_t>(k + 1); i < N1; ++i)
                            x[k * P + j] -= L[i * N1 + k] * x[i * P + j];
                    k--;
                }
                else
                {
                    for (size_t j = 0; j < P; ++j)
                        for (size_t i = static_cast<size_t>(k + 1); i < N1; ++i)
                        {
                            x[k * P + j] -= L[i * N1 + k] * x[i * P + j];
                            x[(k - 1) * P + j] -= L[i * N1 + (k - 1)] * x[i * P + j];
                        }

                    size_t p = static_cast<size_t>(ipiv[k]);
                    if (p != static_cast<size_t>(k))
                        for (size_t j = 0; j < P; ++j)
                            std::swap(x[k * P + j], x[p * P + j]);
                    k -= 2;
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
                lapack_factorize_inplace(N1, L, N1);
                return;
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

                L[i * N1 + i] = std::sqrt(diag_val);
            }
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* L = LLT_mat.data();

            if constexpr (N1 * N1 >= cholesky_lapack_min_size)
            {
                lapack_substitute_inplace(N1, P, L, N1, x, P);
                return;
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

            for (int i = N1 - 1; i >= 0; --i)
            {
                for (size_t j = 0; j < P; ++j)
                {
                    for (size_t k = static_cast<size_t>(i + 1); k < N1; ++k)
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
                lapack_factorize_inplace(N1, N1, LU, N1, ipiv);
                return;
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
                if (pivot_row != k) std::swap_ranges(LU + k * N1, LU + (k + 1) * N1, LU + pivot_row * N1);

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
                lapack_substitute_inplace(N1, P, LU, N1, ipiv, x, P);
                return;
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

            lapack_factorize_inplace(N1, N2, A, U, S, Vt);
        }

        template <size_t P = 1>
        constexpr void substitute_inplace(double* x) const
        {
            const double* U  = U_mat.data();
            const double* Vt = Vt_mat.data();
            const double* S  = S_vec.data();

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
                lapack_factorize_inplace(N1, N2, QR, N2, tau);
                return;
            }

            for (size_t k = 0; k < N2; ++k)
            {
                double norm2 = 0.0;
                for (size_t i = k; i < N1; ++i)
                    norm2 += QR[i * N2 + k] * QR[i * N2 + k];

                double norm = std::sqrt(norm2);
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
                lapack_substitute_inplace(N1, N2, P, QR, N2, tau, x, P);
                return;
            }

            for (size_t k = 0; k < N2; ++k)
            {
                if (std::abs(tau[k]) < linear_tolerance) continue;

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

}
