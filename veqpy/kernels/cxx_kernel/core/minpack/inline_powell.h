#pragma once

#include "cminpack.h"
#include "minpackP.h"

#include <utility>
#include <cstddef>

namespace nonlinear::detail::minpack_inline
{
template <std::size_t N, typename Evaluate>
inline int fdjac1_inline(Evaluate&& evaluate, real* __restrict x, const real* __restrict fvec,
                         real* __restrict fjac, real epsfcn,
                         real* __restrict wa1, real* __restrict wa2)
{
    constexpr int n = static_cast<int>(N);
    constexpr int ldfjac = n;
    constexpr int ml = n - 1;
    constexpr int mu = n - 1;
    /* System generated locals */
    int fjac_dim1, fjac_offset;

    /* Local variables */
    real h;
    int i, j, k;
    real eps, temp;
    int msum;
    real epsmch;
    int iflag = 0;

/*     ********** */

/*     subroutine fdjac1 */

/*     this subroutine computes a forward-difference approximation */
/*     to the n by n jacobian matrix associated with a specified */
/*     problem of n functions in n variables. if the jacobian has */
/*     a banded form, then function evaluations are saved by only */
/*     approximating the nonzero terms. */

/*     the subroutine statement is */

/*       subroutine fdjac1(fcn,n,x,fvec,fjac,ldfjac,iflag,ml,mu,epsfcn, */
/*                         wa1,wa2) */

/*     where */

/*       fcn is the name of the user-supplied subroutine which */
/*         calculates the functions. fcn must be declared */
/*         in an external statement in the user calling */
/*         program, and should be written as follows. */

/*         subroutine fcn(n,x,fvec,iflag) */
/*         integer n,iflag */
/*         double precision x(n),fvec(n) */
/*         ---------- */
/*         calculate the functions at x and */
/*         return this vector in fvec. */
/*         ---------- */
/*         return */
/*         end */

/*         the value of iflag should not be changed by fcn unless */
/*         the user wants to terminate execution of fdjac1. */
/*         in this case set iflag to a negative integer. */

/*       n is a positive integer input variable set to the number */
/*         of functions and variables. */

/*       x is an input array of length n. */

/*       fvec is an input array of length n which must contain the */
/*         functions evaluated at x. */

/*       fjac is an output n by n array which contains the */
/*         approximation to the jacobian matrix evaluated at x. */

/*       ldfjac is a positive integer input variable not less than n */
/*         which specifies the leading dimension of the array fjac. */

/*       iflag is an integer variable which can be used to terminate */
/*         the execution of fdjac1. see description of fcn. */

/*       ml is a nonnegative integer input variable which specifies */
/*         the number of subdiagonals within the band of the */
/*         jacobian matrix. if the jacobian is not banded, set */
/*         ml to at least n - 1. */

/*       epsfcn is an input variable used in determining a suitable */
/*         step length for the forward-difference approximation. this */
/*         approximation assumes that the relative errors in the */
/*         functions are of the order of epsfcn. if epsfcn is less */
/*         than the machine precision, it is assumed that the relative */
/*         errors in the functions are of the order of the machine */
/*         precision. */

/*       mu is a nonnegative integer input variable which specifies */
/*         the number of superdiagonals within the band of the */
/*         jacobian matrix. if the jacobian is not banded, set */
/*         mu to at least n - 1. */

/*       wa1 and wa2 are work arrays of length n. if ml + mu + 1 is at */
/*         least n, then the jacobian is considered dense, and wa2 is */
/*         not referenced. */

/*     subprograms called */

/*       minpack-supplied ... dpmpar */

/*       fortran-supplied ... dabs,dmax1,dsqrt */

/*     argonne national laboratory. minpack project. march 1980. */
/*     burton s. garbow, kenneth e. hillstrom, jorge j. more */

/*     ********** */
    /* Parameter adjustments */
    --wa2;
    --wa1;
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = __cminpack_func__(dpmpar)(1);

    eps = sqrt((max(epsfcn,epsmch)));
    msum = ml + mu + 1;
    if (msum >= n) {

/*        computation of dense approximate jacobian. */

        for (j = 1; j <= n; ++j) {
            temp = x[j];
            h = eps * fabs(temp);
            if (h == 0.) {
                h = eps;
            }
            x[j] = temp + h;
            /* the last parameter of fcn_nn() is set to 2 to differentiate
               calls made to compute the function from calls made to compute
               the Jacobian (see fcn() in examples/hybdrv.c, and how njev
               is used to compute the number of Jacobian evaluations) */
            iflag = evaluate(&x[1], &wa1[1], 2);
            if (iflag < 0) {
                return iflag;
            }
            x[j] = temp;
            for (i = 1; i <= n; ++i) {
                fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h;
            }
        }
        return 0;
    }

/*        computation of banded approximate jacobian. */

    for (k = 1; k <= msum; ++k) {
	for (j = k; j <= n; j += msum) {
	    wa2[j] = x[j];
	    h = eps * fabs(wa2[j]);
	    if (h == 0.) {
		h = eps;
	    }
	    x[j] = wa2[j] + h;
	}
	iflag = evaluate(&x[1], &wa1[1], 1);
	if (iflag < 0) {
            return iflag;
	}
	for (j = k; j <= n; j += msum) {
	    x[j] = wa2[j];
	    h = eps * fabs(wa2[j]);
	    if (h == 0.) {
		h = eps;
	    }
	    for (i = 1; i <= n; ++i) {
		fjac[i + j * fjac_dim1] = 0.;
		if (i >= j - mu && i <= j + ml) {
		    fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h;
		}
	    }
	}
    }
    return 0;

/*     last card of subroutine fdjac1. */

} /* fdjac1_ */





template <std::size_t N, typename Evaluate>
inline int powell_finite_difference(
    Evaluate&& evaluate, real* __restrict x, real* __restrict fvec, real xtol, int maxfev,
    real epsfcn, real* __restrict diag, real factor,
    int* __restrict nfev, real* __restrict fjac, real* __restrict r, real* __restrict qtf,
    real* __restrict wa1, real* __restrict wa2, real* __restrict wa3, real* __restrict wa4)
{
    constexpr int n = static_cast<int>(N);
    constexpr int ml = n - 1;
    constexpr int mu = n - 1;
    constexpr int mode = 1;
    constexpr int nprint = 0;
    constexpr int ldfjac = n;
    constexpr int lr = n * (n + 1) / 2;
    /* Initialized data */

#define p1 ((real).1)
#define p5 ((real).5)
#define p001 .001
#define p0001 1e-4

    /* System generated locals */
    int fjac_dim1, fjac_offset, i1;
    real d1, d2;

    /* Local variables */
    int i, j, l, jm1, iwa[1];
    real sum;
    int sing;
    int iter;
    real temp;
    int msum, iflag;
    real delta = 0.;
    int jeval;
    int ncsuc;
    real ratio;
    real fnorm;
    real pnorm, xnorm = 0., fnorm1;
    int nslow1, nslow2;
    int ncfail;
    real actred, epsmch, prered;
    int info;

/*     ********** */

/*     subroutine hybrd */

/*     the purpose of hybrd is to find a zero of a system of */
/*     n nonlinear functions in n variables by a modification */
/*     of the powell hybrid method. the user must provide a */
/*     subroutine which calculates the functions. the jacobian is */
/*     then calculated by a forward-difference approximation. */

/*     the subroutine statement is */

/*       subroutine hybrd(fcn,n,x,fvec,xtol,maxfev,ml,mu,epsfcn, */
/*                        diag,mode,factor,nprint,info,nfev,fjac, */
/*                        ldfjac,r,lr,qtf,wa1,wa2,wa3,wa4) */

/*     where */

/*       fcn is the name of the user-supplied subroutine which */
/*         calculates the functions. fcn must be declared */
/*         in an external statement in the user calling */
/*         program, and should be written as follows. */

/*         subroutine fcn(n,x,fvec,iflag) */
/*         integer n,iflag */
/*         double precision x(n),fvec(n) */
/*         ---------- */
/*         calculate the functions at x and */
/*         return this vector in fvec. */
/*         --------- */
/*         return */
/*         end */

/*         the value of iflag should not be changed by fcn unless */
/*         the user wants to terminate execution of hybrd. */
/*         in this case set iflag to a negative integer. */

/*       n is a positive integer input variable set to the number */
/*         of functions and variables. */

/*       x is an array of length n. on input x must contain */
/*         an initial estimate of the solution vector. on output x */
/*         contains the final estimate of the solution vector. */

/*       fvec is an output array of length n which contains */
/*         the functions evaluated at the output x. */

/*       xtol is a nonnegative input variable. termination */
/*         occurs when the relative error between two consecutive */
/*         iterates is at most xtol. */

/*       maxfev is a positive integer input variable. termination */
/*         occurs when the number of calls to fcn is at least maxfev */
/*         by the end of an iteration. */

/*       ml is a nonnegative integer input variable which specifies */
/*         the number of subdiagonals within the band of the */
/*         jacobian matrix. if the jacobian is not banded, set */
/*         ml to at least n - 1. */

/*       mu is a nonnegative integer input variable which specifies */
/*         the number of superdiagonals within the band of the */
/*         jacobian matrix. if the jacobian is not banded, set */
/*         mu to at least n - 1. */

/*       epsfcn is an input variable used in determining a suitable */
/*         step length for the forward-difference approximation. this */
/*         approximation assumes that the relative errors in the */
/*         functions are of the order of epsfcn. if epsfcn is less */
/*         than the machine precision, it is assumed that the relative */
/*         errors in the functions are of the order of the machine */
/*         precision. */

/*       diag is an array of length n. if mode = 1 (see */
/*         below), diag is internally set. if mode = 2, diag */
/*         must contain positive entries that serve as */
/*         multiplicative scale factors for the variables. */

/*       mode is an integer input variable. if mode = 1, the */
/*         variables will be scaled internally. if mode = 2, */
/*         the scaling is specified by the input diag. other */
/*         values of mode are equivalent to mode = 1. */

/*       factor is a positive input variable used in determining the */
/*         initial step bound. this bound is set to the product of */
/*         factor and the euclidean norm of diag*x if nonzero, or else */
/*         to factor itself. in most cases factor should lie in the */
/*         interval (.1,100.). 100. is a generally recommended value. */

/*       nprint is an integer input variable that enables controlled */
/*         printing of iterates if it is positive. in this case, */
/*         fcn is called with iflag = 0 at the beginning of the first */
/*         iteration and every nprint iterations thereafter and */
/*         immediately prior to return, with x and fvec available */
/*         for printing. if nprint is not positive, no special calls */
/*         of fcn with iflag = 0 are made. */

/*       info is an integer output variable. if the user has */
/*         terminated execution, info is set to the (negative) */
/*         value of iflag. see description of fcn. otherwise, */
/*         info is set as follows. */

/*         info = 0   improper input parameters. */

/*         info = 1   relative error between two consecutive iterates */
/*                    is at most xtol. */

/*         info = 2   number of calls to fcn has reached or exceeded */
/*                    maxfev. */

/*         info = 3   xtol is too small. no further improvement in */
/*                    the approximate solution x is possible. */

/*         info = 4   iteration is not making good progress, as */
/*                    measured by the improvement from the last */
/*                    five jacobian evaluations. */

/*         info = 5   iteration is not making good progress, as */
/*                    measured by the improvement from the last */
/*                    ten iterations. */

/*       nfev is an integer output variable set to the number of */
/*         calls to fcn. */

/*       fjac is an output n by n array which contains the */
/*         orthogonal matrix q produced by the qr factorization */
/*         of the final approximate jacobian. */

/*       ldfjac is a positive integer input variable not less than n */
/*         which specifies the leading dimension of the array fjac. */

/*       r is an output array of length lr which contains the */
/*         upper triangular matrix produced by the qr factorization */
/*         of the final approximate jacobian, stored rowwise. */

/*       lr is a positive integer input variable not less than */
/*         (n*(n+1))/2. */

/*       qtf is an output array of length n which contains */
/*         the vector (q transpose)*fvec. */

/*       wa1, wa2, wa3, and wa4 are work arrays of length n. */

/*     subprograms called */

/*       user-supplied ...... fcn */

/*       minpack-supplied ... dogleg,dpmpar,enorm,fdjac1, */
/*                            qform,qrfac,r1mpyq,r1updt */

/*       fortran-supplied ... dabs,dmax1,dmin1,min0,mod */

/*     argonne national laboratory. minpack project. march 1980. */
/*     burton s. garbow, kenneth e. hillstrom, jorge j. more */

/*     ********** */
    /* Parameter adjustments */
    --wa4;
    --wa3;
    --wa2;
    --wa1;
    --qtf;
    --diag;
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;
    --r;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = __cminpack_func__(dpmpar)(1);

    info = 0;
    iflag = 0;
    *nfev = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || xtol < 0. || maxfev <= 0 || ml < 0 || mu < 0 ||
	    factor <= 0. || ldfjac < n || lr < n * (n + 1) / 2) {
	goto TERMINATE;
    }
    if (mode == 2) {
        for (j = 1; j <= n; ++j) {
            if (diag[j] <= 0.) {
                goto TERMINATE;
            }
        }
    }

/*     evaluate the function at the starting point */
/*     and calculate its norm. */

    iflag = evaluate(&x[1], &fvec[1], 1);
    *nfev = 1;
    if (iflag < 0) {
	goto TERMINATE;
    }
    fnorm = __cminpack_func__(enorm)(n, &fvec[1]);

/*     determine the number of calls to fcn needed to compute */
/*     the jacobian matrix. */

/* Computing MIN */
    i1 = ml + mu + 1;
    msum = min(i1,n);

/*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

/*     beginning of the outer loop. */

    for (;;) {
        jeval = TRUE_;

/*        calculate the jacobian matrix. */

        iflag = fdjac1_inline<N>(evaluate, &x[1], &fvec[1], &fjac[fjac_offset],
                       epsfcn, &wa1[1], &wa2[1]);
        *nfev += msum;
        if (iflag < 0) {
            goto TERMINATE;
        }

/*        compute the qr factorization of the jacobian. */

        __cminpack_func__(qrfac)(n, n, &fjac[fjac_offset], ldfjac, FALSE_, iwa, 1,
              &wa1[1], &wa2[1], &wa3[1]);

/*        on the first iteration and if mode is 1, scale according */
/*        to the norms of the columns of the initial jacobian. */

        if (iter == 1) {
            if (mode != 2) {
                for (j = 1; j <= n; ++j) {
                    diag[j] = wa2[j];
                    if (wa2[j] == 0.) {
                        diag[j] = 1.;
                    }
                }
            }

/*        on the first iteration, calculate the norm of the scaled x */
/*        and initialize the step bound delta. */

            for (j = 1; j <= n; ++j) {
                wa3[j] = diag[j] * x[j];
            }
            xnorm = __cminpack_func__(enorm)(n, &wa3[1]);
            delta = factor * xnorm;
            if (delta == 0.) {
                delta = factor;
            }
        }

/*        form (q transpose)*fvec and store in qtf. */

        for (i = 1; i <= n; ++i) {
            qtf[i] = fvec[i];
        }
        for (j = 1; j <= n; ++j) {
            if (fjac[j + j * fjac_dim1] != 0.) {
                sum = 0.;
                for (i = j; i <= n; ++i) {
                    sum += fjac[i + j * fjac_dim1] * qtf[i];
                }
                temp = -sum / fjac[j + j * fjac_dim1];
                for (i = j; i <= n; ++i) {
                    qtf[i] += fjac[i + j * fjac_dim1] * temp;
                }
            }
        }

/*        copy the triangular factor of the qr factorization into r. */

        sing = FALSE_;
        for (j = 1; j <= n; ++j) {
            l = j;
            jm1 = j - 1;
            if (jm1 >= 1) {
                for (i = 1; i <= jm1; ++i) {
                    r[l] = fjac[i + j * fjac_dim1];
                    l = l + n - i;
                }
            }
            r[l] = wa1[j];
            if (wa1[j] == 0.) {
                sing = TRUE_;
            }
        }

/*        accumulate the orthogonal factor in fjac. */

        __cminpack_func__(qform)(n, n, &fjac[fjac_offset], ldfjac, &wa1[1]);

/*        rescale if necessary. */

        if (mode != 2) {
            for (j = 1; j <= n; ++j) {
                /* Computing MAX */
                d1 = diag[j], d2 = wa2[j];
                diag[j] = max(d1,d2);
            }
        }

/*        beginning of the inner loop. */

        for (;;) {

/*           if requested, call fcn to enable printing of iterates. */

            if (nprint > 0) {
                iflag = 0;
                if ((iter - 1) % nprint == 0) {
                    iflag = evaluate(&x[1], &fvec[1], 0);
                }
                if (iflag < 0) {
                    goto TERMINATE;
                }
            }

/*           determine the direction p. */

            __cminpack_func__(dogleg)(n, &r[1], lr, &diag[1], &qtf[1], delta, &wa1[1], &wa2[1], &wa3[1]);

/*           store the direction p and x + p. calculate the norm of p. */

            for (j = 1; j <= n; ++j) {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            pnorm = __cminpack_func__(enorm)(n, &wa3[1]);

/*           on the first iteration, adjust the initial step bound. */

            if (iter == 1) {
                delta = min(delta,pnorm);
            }

/*           evaluate the function at x + p and calculate its norm. */

            iflag = evaluate(&wa2[1], &wa4[1], 1);
            ++(*nfev);
            if (iflag < 0) {
                goto TERMINATE;
            }
            fnorm1 = __cminpack_func__(enorm)(n, &wa4[1]);

/*           compute the scaled actual reduction. */

            actred = -1.;
            if (fnorm1 < fnorm) {
                /* Computing 2nd power */
                d1 = fnorm1 / fnorm;
                actred = 1 - d1 * d1;
            }

/*           compute the scaled predicted reduction. */

            l = 1;
            for (i = 1; i <= n; ++i) {
                sum = 0.;
                for (j = i; j <= n; ++j) {
                    sum += r[l] * wa1[j];
                    ++l;
                }
                wa3[i] = qtf[i] + sum;
            }
            temp = __cminpack_func__(enorm)(n, &wa3[1]);
            prered = 0.;
            if (temp < fnorm) {
                /* Computing 2nd power */
                d1 = temp / fnorm;
                prered = 1 - d1 * d1;
            }

/*           compute the ratio of the actual to the predicted */
/*           reduction. */

            ratio = 0.;
            if (prered > 0.) {
                ratio = actred / prered;
            }

/*           update the step bound. */

            if (ratio < p1) {
                ncsuc = 0;
                ++ncfail;
                delta = p5 * delta;
            } else {
                ncfail = 0;
                ++ncsuc;
                if (ratio >= p5 || ncsuc > 1) {
                    /* Computing MAX */
                    d1 = pnorm / p5;
                    delta = max(delta,d1);
                }
                if (fabs(ratio - 1) <= p1) {
                    delta = pnorm / p5;
                }
            }

/*           test for successful iteration. */

            if (ratio >= p0001) {

/*           successful iteration. update x, fvec, and their norms. */

                for (j = 1; j <= n; ++j) {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                    fvec[j] = wa4[j];
                }
                xnorm = __cminpack_func__(enorm)(n, &wa2[1]);
                fnorm = fnorm1;
                ++iter;
            }

/*           determine the progress of the iteration. */

            ++nslow1;
            if (actred >= p001) {
                nslow1 = 0;
            }
            if (jeval) {
                ++nslow2;
            }
            if (actred >= p1) {
                nslow2 = 0;
            }

/*           test for convergence. */

            if (delta <= xtol * xnorm || fnorm == 0.) {
                info = 1;
            }
            if (info != 0) {
                goto TERMINATE;
            }

/*           tests for termination and stringent tolerances. */

            if (*nfev >= maxfev) {
                info = 2;
            }
            /* Computing MAX */
            d1 = p1 * delta;
            if (p1 * max(d1,pnorm) <= epsmch * xnorm) {
                info = 3;
            }
            if (nslow2 == 5) {
                info = 4;
            }
            if (nslow1 == 10) {
                info = 5;
            }
            if (info != 0) {
                goto TERMINATE;
            }

/*           criterion for recalculating jacobian approximation */
/*           by forward differences. */

            if (ncfail == 2) {
                goto TERMINATE_INNER_LOOP;
            }

/*           calculate the rank one modification to the jacobian */
/*           and update qtf if necessary. */

            for (j = 1; j <= n; ++j) {
                sum = 0.;
                for (i = 1; i <= n; ++i) {
                    sum += fjac[i + j * fjac_dim1] * wa4[i];
                }
                wa2[j] = (sum - wa3[j]) / pnorm;
                wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
                if (ratio >= p0001) {
                    qtf[j] = sum;
                }
            }

/*           compute the qr factorization of the updated jacobian. */

            __cminpack_func__(r1updt)(n, n, &r[1], lr, &wa1[1], &wa2[1], &wa3[1], &sing);
            __cminpack_func__(r1mpyq)(n, n, &fjac[fjac_offset], ldfjac, &wa2[1], &wa3[1]);
            __cminpack_func__(r1mpyq)(1, n, &qtf[1], 1, &wa2[1], &wa3[1]);

/*           end of the inner loop. */

            jeval = FALSE_;
        }
TERMINATE_INNER_LOOP:
        ;
/*        end of the outer loop. */

    }
TERMINATE:

/*     termination, either normal or user imposed. */

    if (iflag < 0) {
	info = iflag;
    }
    if (nprint > 0) {
	evaluate(&x[1], &fvec[1], 0);
    }
    return info;

/*     last card of subroutine hybrd. */

} /* hybrd_ */





template <std::size_t N, typename Evaluate>
inline int powell_with_jacobian(
    Evaluate&& evaluate, real* __restrict x, real* __restrict fvec, real* __restrict fjac,
    real xtol, int maxfev, real* __restrict diag, real factor,
    int* __restrict nfev, int* __restrict njev, real* __restrict r, real* __restrict qtf,
    real* __restrict wa1, real* __restrict wa2, real* __restrict wa3, real* __restrict wa4)
{
    constexpr int n = static_cast<int>(N);
    constexpr int mode = 1;
    constexpr int nprint = 0;
    constexpr int ldfjac = n;
    constexpr int lr = n * (n + 1) / 2;
    /* Initialized data */

#define p1 ((real).1)
#define p5 ((real).5)
#define p001 .001
#define p0001 1e-4

    /* System generated locals */
    int fjac_dim1, fjac_offset;
    real d1, d2;

    /* Local variables */
    int i, j, l, jm1, iwa[1];
    real sum;
    int sing;
    int iter;
    real temp;
    int iflag;
    real delta = 0.;
    int jeval;
    int ncsuc;
    real ratio;
    real fnorm;
    real pnorm, xnorm = 0., fnorm1;
    int nslow1, nslow2;
    int ncfail;
    real actred, epsmch, prered;
    int info;

/*     ********** */

/*     subroutine hybrj */

/*     the purpose of hybrj is to find a zero of a system of */
/*     n nonlinear functions in n variables by a modification */
/*     of the powell hybrid method. the user must provide a */
/*     subroutine which calculates the functions and the jacobian. */

/*     the subroutine statement is */

/*       subroutine hybrj(fcn,n,x,fvec,fjac,ldfjac,xtol,maxfev,diag, */
/*                        mode,factor,nprint,info,nfev,njev,r,lr,qtf, */
/*                        wa1,wa2,wa3,wa4) */

/*     where */

/*       fcn is the name of the user-supplied subroutine which */
/*         calculates the functions and the jacobian. fcn must */
/*         be declared in an external statement in the user */
/*         calling program, and should be written as follows. */

/*         subroutine fcn(n,x,fvec,fjac,ldfjac,iflag) */
/*         integer n,ldfjac,iflag */
/*         double precision x(n),fvec(n),fjac(ldfjac,n) */
/*         ---------- */
/*         if iflag = 1 calculate the functions at x and */
/*         return this vector in fvec. do not alter fjac. */
/*         if iflag = 2 calculate the jacobian at x and */
/*         return this matrix in fjac. do not alter fvec. */
/*         --------- */
/*         return */
/*         end */

/*         the value of iflag should not be changed by fcn unless */
/*         the user wants to terminate execution of hybrj. */
/*         in this case set iflag to a negative integer. */

/*       n is a positive integer input variable set to the number */
/*         of functions and variables. */

/*       x is an array of length n. on input x must contain */
/*         an initial estimate of the solution vector. on output x */
/*         contains the final estimate of the solution vector. */

/*       fvec is an output array of length n which contains */
/*         the functions evaluated at the output x. */

/*       fjac is an output n by n array which contains the */
/*         orthogonal matrix q produced by the qr factorization */
/*         of the final approximate jacobian. */

/*       ldfjac is a positive integer input variable not less than n */
/*         which specifies the leading dimension of the array fjac. */

/*       xtol is a nonnegative input variable. termination */
/*         occurs when the relative error between two consecutive */
/*         iterates is at most xtol. */

/*       maxfev is a positive integer input variable. termination */
/*         occurs when the number of calls to fcn with iflag = 1 */
/*         has reached maxfev. */

/*       diag is an array of length n. if mode = 1 (see */
/*         below), diag is internally set. if mode = 2, diag */
/*         must contain positive entries that serve as */
/*         multiplicative scale factors for the variables. */

/*       mode is an integer input variable. if mode = 1, the */
/*         variables will be scaled internally. if mode = 2, */
/*         the scaling is specified by the input diag. other */
/*         values of mode are equivalent to mode = 1. */

/*       factor is a positive input variable used in determining the */
/*         initial step bound. this bound is set to the product of */
/*         factor and the euclidean norm of diag*x if nonzero, or else */
/*         to factor itself. in most cases factor should lie in the */
/*         interval (.1,100.). 100. is a generally recommended value. */

/*       nprint is an integer input variable that enables controlled */
/*         printing of iterates if it is positive. in this case, */
/*         fcn is called with iflag = 0 at the beginning of the first */
/*         iteration and every nprint iterations thereafter and */
/*         immediately prior to return, with x and fvec available */
/*         for printing. fvec and fjac should not be altered. */
/*         if nprint is not positive, no special calls of fcn */
/*         with iflag = 0 are made. */

/*       info is an integer output variable. if the user has */
/*         terminated execution, info is set to the (negative) */
/*         value of iflag. see description of fcn. otherwise, */
/*         info is set as follows. */

/*         info = 0   improper input parameters. */

/*         info = 1   relative error between two consecutive iterates */
/*                    is at most xtol. */

/*         info = 2   number of calls to fcn with iflag = 1 has */
/*                    reached maxfev. */

/*         info = 3   xtol is too small. no further improvement in */
/*                    the approximate solution x is possible. */

/*         info = 4   iteration is not making good progress, as */
/*                    measured by the improvement from the last */
/*                    five jacobian evaluations. */

/*         info = 5   iteration is not making good progress, as */
/*                    measured by the improvement from the last */
/*                    ten iterations. */

/*       nfev is an integer output variable set to the number of */
/*         calls to fcn with iflag = 1. */

/*       njev is an integer output variable set to the number of */
/*         calls to fcn with iflag = 2. */

/*       r is an output array of length lr which contains the */
/*         upper triangular matrix produced by the qr factorization */
/*         of the final approximate jacobian, stored rowwise. */

/*       lr is a positive integer input variable not less than */
/*         (n*(n+1))/2. */

/*       qtf is an output array of length n which contains */
/*         the vector (q transpose)*fvec. */

/*       wa1, wa2, wa3, and wa4 are work arrays of length n. */

/*     subprograms called */

/*       user-supplied ...... fcn */

/*       minpack-supplied ... dogleg,dpmpar,enorm, */
/*                            qform,qrfac,r1mpyq,r1updt */

/*       fortran-supplied ... dabs,dmax1,dmin1,mod */

/*     argonne national laboratory. minpack project. march 1980. */
/*     burton s. garbow, kenneth e. hillstrom, jorge j. more */

/*     ********** */
    /* Parameter adjustments */
    --wa4;
    --wa3;
    --wa2;
    --wa1;
    --qtf;
    --diag;
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;
    --r;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = __cminpack_func__(dpmpar)(1);

    info = 0;
    iflag = 0;
    *nfev = 0;
    *njev = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || ldfjac < n || xtol < 0. || maxfev <= 0 || factor <= 
	    0. || lr < n * (n + 1) / 2) {
	goto TERMINATE;
    }
    if (mode == 2) {
        for (j = 1; j <= n; ++j) {
            if (diag[j] <= 0.) {
                goto TERMINATE;
            }
        }
    }

/*     evaluate the function at the starting point */
/*     and calculate its norm. */

    iflag = evaluate(&x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 1);
    *nfev = 1;
    if (iflag < 0) {
	goto TERMINATE;
    }
    fnorm = __cminpack_func__(enorm)(n, &fvec[1]);

/*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

/*     beginning of the outer loop. */

    for (;;) {
        jeval = TRUE_;

/*        calculate the jacobian matrix. */

        iflag = evaluate(&x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 2);
        ++(*njev);
        if (iflag < 0) {
            goto TERMINATE;
        }

/*        compute the qr factorization of the jacobian. */

        __cminpack_func__(qrfac)(n, n, &fjac[fjac_offset], ldfjac, FALSE_, iwa, 1,
              &wa1[1], &wa2[1], &wa3[1]);

/*        on the first iteration and if mode is 1, scale according */
/*        to the norms of the columns of the initial jacobian. */

        if (iter == 1) {
            if (mode != 2) {
                for (j = 1; j <= n; ++j) {
                    diag[j] = wa2[j];
                    if (wa2[j] == 0.) {
                        diag[j] = 1.;
                    }
                }
            }

/*        on the first iteration, calculate the norm of the scaled x */
/*        and initialize the step bound delta. */

            for (j = 1; j <= n; ++j) {
                wa3[j] = diag[j] * x[j];
            }
            xnorm = __cminpack_func__(enorm)(n, &wa3[1]);
            delta = factor * xnorm;
            if (delta == 0.) {
                delta = factor;
            }
        }

/*        form (q transpose)*fvec and store in qtf. */

        for (i = 1; i <= n; ++i) {
            qtf[i] = fvec[i];
        }
        for (j = 1; j <= n; ++j) {
            if (fjac[j + j * fjac_dim1] != 0.) {
                sum = 0.;
                for (i = j; i <= n; ++i) {
                    sum += fjac[i + j * fjac_dim1] * qtf[i];
                }
                temp = -sum / fjac[j + j * fjac_dim1];
                for (i = j; i <= n; ++i) {
                    qtf[i] += fjac[i + j * fjac_dim1] * temp;
                }
            }
        }

/*        copy the triangular factor of the qr factorization into r. */

        sing = FALSE_;
        for (j = 1; j <= n; ++j) {
            l = j;
            jm1 = j - 1;
            if (jm1 >= 1) {
                for (i = 1; i <= jm1; ++i) {
                    r[l] = fjac[i + j * fjac_dim1];
                    l = l + n - i;
                }
            }
            r[l] = wa1[j];
            if (wa1[j] == 0.) {
                sing = TRUE_;
            }
        }

/*        accumulate the orthogonal factor in fjac. */

        __cminpack_func__(qform)(n, n, &fjac[fjac_offset], ldfjac, &wa1[1]);

/*        rescale if necessary. */

        if (mode != 2) {
            for (j = 1; j <= n; ++j) {
                /* Computing MAX */
                d1 = diag[j], d2 = wa2[j];
                diag[j] = max(d1,d2);
            }
        }

/*        beginning of the inner loop. */

        for (;;) {

/*           if requested, call fcn to enable printing of iterates. */

            if (nprint > 0) {
                iflag = 0;
                if ((iter - 1) % nprint == 0) {
                    iflag = evaluate(&x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 0);
                }
                if (iflag < 0) {
                    goto TERMINATE;
                }
            }

/*           determine the direction p. */

            __cminpack_func__(dogleg)(n, &r[1], lr, &diag[1], &qtf[1], delta, &wa1[1], &wa2[1], &wa3[1]);

/*           store the direction p and x + p. calculate the norm of p. */

            for (j = 1; j <= n; ++j) {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            pnorm = __cminpack_func__(enorm)(n, &wa3[1]);

/*           on the first iteration, adjust the initial step bound. */

            if (iter == 1) {
                delta = min(delta,pnorm);
            }

/*           evaluate the function at x + p and calculate its norm. */

            iflag = evaluate(&wa2[1], &wa4[1], &fjac[fjac_offset], ldfjac, 1);
            ++(*nfev);
            if (iflag < 0) {
                goto TERMINATE;
            }
            fnorm1 = __cminpack_func__(enorm)(n, &wa4[1]);

/*           compute the scaled actual reduction. */

            actred = -1.;
            if (fnorm1 < fnorm) {
                /* Computing 2nd power */
                d1 = fnorm1 / fnorm;
                actred = 1 - d1 * d1;
            }

/*           compute the scaled predicted reduction. */

            l = 1;
            for (i = 1; i <= n; ++i) {
                sum = 0.;
                for (j = i; j <= n; ++j) {
                    sum += r[l] * wa1[j];
                    ++l;
                }
                wa3[i] = qtf[i] + sum;
            }
            temp = __cminpack_func__(enorm)(n, &wa3[1]);
            prered = 0.;
            if (temp < fnorm) {
                /* Computing 2nd power */
                d1 = temp / fnorm;
                prered = 1 - d1 * d1;
            }

/*           compute the ratio of the actual to the predicted */
/*           reduction. */

            ratio = 0.;
            if (prered > 0.) {
                ratio = actred / prered;
            }

/*           update the step bound. */

            if (ratio < p1) {
                ncsuc = 0;
                ++ncfail;
                delta = p5 * delta;
            } else {
                ncfail = 0;
                ++ncsuc;
                if (ratio >= p5 || ncsuc > 1) {
                    /* Computing MAX */
                    d1 = pnorm / p5;
                    delta = max(delta,d1);
                }
                if (fabs(ratio - 1) <= p1) {
                    delta = pnorm / p5;
                }
            }

/*           test for successful iteration. */

            if (ratio >= p0001) {

/*           successful iteration. update x, fvec, and their norms. */

                for (j = 1; j <= n; ++j) {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                    fvec[j] = wa4[j];
                }
                xnorm = __cminpack_func__(enorm)(n, &wa2[1]);
                fnorm = fnorm1;
                ++iter;
            }

/*           determine the progress of the iteration. */

            ++nslow1;
            if (actred >= p001) {
                nslow1 = 0;
            }
            if (jeval) {
                ++nslow2;
            }
            if (actred >= p1) {
                nslow2 = 0;
            }

/*           test for convergence. */

            if (delta <= xtol * xnorm || fnorm == 0.) {
                info = 1;
            }
            if (info != 0) {
                goto TERMINATE;
            }

/*           tests for termination and stringent tolerances. */

            if (*nfev >= maxfev) {
                info = 2;
            }
            /* Computing MAX */
            d1 = p1 * delta;
            if (p1 * max(d1,pnorm) <= epsmch * xnorm) {
                info = 3;
            }
            if (nslow2 == 5) {
                info = 4;
            }
            if (nslow1 == 10) {
                info = 5;
            }
            if (info != 0) {
                goto TERMINATE;
            }

/*           criterion for recalculating jacobian. */

            if (ncfail == 2) {
                goto TERMINATE_INNER_LOOP;
            }

/*           calculate the rank one modification to the jacobian */
/*           and update qtf if necessary. */

            for (j = 1; j <= n; ++j) {
                sum = 0.;
                for (i = 1; i <= n; ++i) {
                    sum += fjac[i + j * fjac_dim1] * wa4[i];
                }
                wa2[j] = (sum - wa3[j]) / pnorm;
                wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
                if (ratio >= p0001) {
                    qtf[j] = sum;
                }
            }

/*           compute the qr factorization of the updated jacobian. */

            __cminpack_func__(r1updt)(n, n, &r[1], lr, &wa1[1], &wa2[1], &wa3[1], &sing);
            __cminpack_func__(r1mpyq)(n, n, &fjac[fjac_offset], ldfjac, &wa2[1], &wa3[1]);
            __cminpack_func__(r1mpyq)(1, n, &qtf[1], 1, &wa2[1], &wa3[1]);

/*           end of the inner loop. */

            jeval = FALSE_;
        }
TERMINATE_INNER_LOOP:
        ;
/*        end of the outer loop. */

    }
TERMINATE:

/*     termination, either normal or user imposed. */

    if (iflag < 0) {
	info = iflag;
    }
    if (nprint > 0) {
	evaluate(&x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 0);
    }
    return info;

/*     last card of subroutine hybrj. */

} /* hybrj_ */




} // namespace nonlinear::detail::minpack_inline

#include "inline_lm.h"

#undef real
#undef min
#undef max
#undef abs
#undef TRUE_
#undef FALSE_
