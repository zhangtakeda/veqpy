// Runtime bridges to LAPACKE's blocked dense kernels.

#include "linalg.h"
#include <algorithm>
#include <cassert>
#include <cblas.h>
#include <lapacke.h>

static_assert(sizeof(lapack_int) == sizeof(int),
              "lapack_int size mismatch; disable runtime LAPACKE dispatch for this ABI");

[[maybe_unused]] const bool blas_runtime_configured = []
{
#if defined(VEQPY_CXX_FORCE_OPENBLAS_SINGLE_THREAD)
    openblas_set_num_threads(1);
#endif
    return true;
}();

namespace linalg::detail
{
    int Doolittle::lapack_factorize_inplace(int m, int n, double* a, int lda, int* ipiv)
    {
        return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
    }

    void
    Doolittle::lapack_substitute_inplace(int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb)
    {
        const int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, nrhs, a, lda, ipiv, b, ldb);
        assert(info == 0);
    }

    int Cholesky::lapack_factorize_inplace(int n, double* a, int lda)
    {
        return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, a, lda);
    }

    void Cholesky::lapack_substitute_inplace(int n, int nrhs, const double* a, int lda, double* b, int ldb)
    {
        const int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, a, lda, b, ldb);
        assert(info == 0);
    }

    int BunchKaufman::lapack_factorize_inplace(int n, double* a, int lda, int* ipiv)
    {
        return LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', n, a, lda, ipiv);
    }

    void BunchKaufman::lapack_substitute_inplace(
        int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb)
    {
        const int info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, a, lda, ipiv, b, ldb);
        assert(info == 0);
    }

    int Householder::lapack_factorize_inplace(int m, int n, double* a, int lda, double* tau)
    {
        return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
    }

    void Householder::lapack_substitute_inplace(
        int m, int n, int nrhs, const double* a, int lda, const double* tau, double* c, int ldc)
    {
        const int reflector_info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', m, nrhs, n, a, lda, tau, c, ldc);
        assert(reflector_info == 0);

        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, nrhs, 1.0, a, lda, c, ldc);
    }

    int GolubReinsch::lapack_factorize_inplace(
        int m, int n, const double* source, double* work, double* u, double* s, double* vt)
    {
        std::copy(source, source + m * n, work);
        return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, work, n, s, u, m, vt, n);
    }

    void GolubReinsch::lapack_substitute_inplace(
        int m, int n, const double* u, const double* s, const double* vt, double* b, int nrhs, double* work)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, nrhs, m, 1.0, u, m, b, nrhs, 0.0, work, nrhs);

        const int rank = std::min(m, n);
        for (int i = 0; i < rank; ++i)
        {
            const double inverse = s[i] > 1.0e-12 ? 1.0 / s[i] : 0.0;
            for (int rhs = 0; rhs < nrhs; ++rhs)
                work[i * nrhs + rhs] *= inverse;
        }

        if (n > rank)
            std::fill(work + rank * nrhs, work + n * nrhs, 0.0);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, nrhs, n, 1.0, vt, n, work, nrhs, 0.0, b, nrhs);
    }
} // namespace linalg::detail
