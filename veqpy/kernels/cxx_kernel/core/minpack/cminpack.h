#pragma once
#define __CMINPACK_H__

// Minimal declarations retained for the MINPACK-derived numerical primitives.
// Solver state machines and residual evaluation live in the templated C++ headers.

#define __cminpack_real__ double
#define __cminpack_double__
#define __cminpack_attr__
#define CMINPACK_EXPORT
#define __cminpack_func__(function) veqpy_minpack_ ## function
#define __cminpack_blas__(function) d##function##_
#define __cminpack_lapack__(function) d##function

double __cminpack_func__(dpmpar)(int i);
double __cminpack_func__(enorm)(int n, const double* x);
void __cminpack_func__(dogleg)(int n, const double* r, int lr, const double* diag, const double* qtb,
                               double delta, double* x, double* wa1, double* wa2);
void __cminpack_func__(qrfac_apply_qt)(int m, int n, double* a, int lda, int pivot, int* ipvt,
                                       int lipvt, double* rdiag, double* acnorm, double* wa,
                                       double* qtf, double* packed_r, int* singular);
void __cminpack_func__(qrsolv_isotropic)(int n, double* r, int ldr, const int* ipvt,
                                         double diagonal, const double* qtb, double* x,
                                         double* sdiag, double* wa);
void __cminpack_func__(qform)(int m, int n, double* q, int ldq, double* wa);
void __cminpack_func__(r1updt)(int m, int n, double* s, int ls, const double* u, double* v,
                               double* w, int* sing);
void __cminpack_func__(r1mpyq_pair)(int n, double* q, int ldq, double* qtf,
                                    const double* v, const double* w);
void __cminpack_func__(lmpar_unit)(int n, double* r, int ldr, const int* ipvt,
                                   const double* qtb, double delta, double* par, double* x,
                                   double* sdiag, double* wa1, double* wa2);
