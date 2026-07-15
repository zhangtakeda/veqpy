#pragma once
#define __CMINPACK_H__

// Minimal declarations retained for the MINPACK-derived numerical primitives.
// Solver state machines and residual evaluation live in the templated C++ headers.

#define __cminpack_real__ double
#define __cminpack_double__
#define __cminpack_attr__
#define CMINPACK_EXPORT
#define __cminpack_func__(function) function
#define __cminpack_blas__(function) d##function##_
#define __cminpack_lapack__(function) d##function

double dpmpar(int i);
double enorm(int n, const double* x);
void dogleg(int n, const double* r, int lr, const double* diag, const double* qtb,
            double delta, double* x, double* wa1, double* wa2);
void qrfac_apply_qt(int m, int n, double* a, int lda, int pivot, int* ipvt,
                    int lipvt, double* rdiag, double* acnorm, double* wa,
                    double* qtf);
void qrsolv(int n, double* r, int ldr, const int* ipvt, const double* diag,
            const double* qtb, double* x, double* sdiag, double* wa);
void qform(int m, int n, double* q, int ldq, double* wa);
void r1updt(int m, int n, double* s, int ls, const double* u, double* v,
            double* w, int* sing);
void r1mpyq(int m, int n, double* a, int lda, const double* v, const double* w);
void lmpar(int n, double* r, int ldr, const int* ipvt, const double* diag,
           const double* qtb, double delta, double* par, double* x,
           double* sdiag, double* wa1, double* wa2);
