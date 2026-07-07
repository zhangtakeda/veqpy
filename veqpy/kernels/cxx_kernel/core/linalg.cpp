// BLAS/LAPACK-backed dense linear-algebra implementations for generated Cxx Kernel artifacts.

#include "linalg.h"
#include <cblas.h>
#include <lapacke.h>

static_assert(sizeof(lapack_int) == sizeof(int),
              "lapack_int size mismatch, set lapack_threshold to a very large value");

[[maybe_unused]] const bool blas_runtime_configured = []
{
#if defined(VEQLIB_FORCE_OPENBLAS_SINGLE_THREAD)
    openblas_set_num_threads(1);
#endif
    return true;
}();

namespace linalg::detail
{
    void Doolittle::lapack_factorize_inplace(int m, int n, double* a, int lda, int* ipiv)
    {
        [[maybe_unused]] int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
        assert(info == 0);
    }

    void
    Doolittle::lapack_substitute_inplace(int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb)
    {
        [[maybe_unused]] int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, nrhs, a, lda, ipiv, b, ldb);
        assert(info == 0);
    }

    void Cholesky::lapack_factorize_inplace(int n, double* a, int lda)
    {
        [[maybe_unused]] int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, a, lda);
        assert(info == 0);
    }

    void Cholesky::lapack_substitute_inplace(int n, int nrhs, const double* a, int lda, double* b, int ldb)
    {
        [[maybe_unused]] int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, a, lda, b, ldb);
        assert(info == 0);
    }

    void BunchKaufman::lapack_factorize_inplace(int n, double* a, int lda, int* ipiv)
    {
        [[maybe_unused]] int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', n, a, lda, ipiv);
        assert(info == 0);
    }

    void BunchKaufman::lapack_substitute_inplace(
        int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb)
    {
        [[maybe_unused]] int info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, a, lda, ipiv, b, ldb);
        assert(info == 0);
    }

    void Householder::lapack_factorize_inplace(int m, int n, double* a, int lda, double* tau)
    {
        [[maybe_unused]] int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
        assert(info == 0);
    }

    void Householder::lapack_substitute_inplace(
        int m, int n, int nrhs, const double* a, int lda, const double* tau, double* c, int ldc)
    {
        [[maybe_unused]] int info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', m, nrhs, n, a, lda, tau, c, ldc);
        assert(info == 0);

        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, nrhs, 1.0, a, lda, c, ldc);
    }

    void GolubReinsch::lapack_factorize_inplace(int m, int n, const double* a, double* u, double* s, double* vt)
    {
        double* buffer = new double[static_cast<std::size_t>(m * n)];
        std::copy(a, a + m * n, buffer);
        [[maybe_unused]] int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, buffer, n, s, u, m, vt, n);
        delete[] buffer;
        assert(info == 0);
    }

    void GolubReinsch::lapack_substitute_inplace(
        int m, int n, const double* u, const double* s, const double* vt, double* b, int nrhs)
    {
        double* buffer = new double[static_cast<std::size_t>(std::max(m, n) * nrhs)];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, nrhs, m, 1.0, u, m, b, nrhs, 0.0, buffer, nrhs);

        int rank = std::min(m, n);
        for (int i = 0; i < rank; ++i)
        {
            double sinv = (s[i] > 1e-12) ? 1.0 / s[i] : 0.0;
            for (int p = 0; p < nrhs; ++p)
                buffer[i * nrhs + p] *= sinv;
        }

        if (n > rank)
            std::fill(buffer + rank * nrhs, buffer + n * nrhs, 0.0);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, nrhs, n, 1.0, vt, n, buffer, nrhs, 0.0, b, nrhs);

        delete[] buffer;
    }
} // namespace linalg::detail
