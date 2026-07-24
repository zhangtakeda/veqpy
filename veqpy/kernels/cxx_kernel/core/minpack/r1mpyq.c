/* r1mpyq.f -- translated by f2c (version 20020621).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "cminpack.h"
#include <math.h>
#include "cminpackP.h"

__cminpack_attr__
void __cminpack_func__(r1mpyq_pair)(int n, real *q, int
	ldq, real *qtf, const real *v, const real *w)
{
    /* System generated locals */
    int q_dim1, q_offset;

    /* Local variables */
    int i, j, nm1, nmj;
    real cos, sin, temp;

/*     ********** */

/*     subroutine r1mpyq_pair */

/*     given an n by n matrix a and an n-vector b, this subroutine */
/*     computes a*q and b*q where */
/*     q is the product of 2*(n - 1) transformations */

/*           gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1) */

/*     and gv(i), gw(i) are givens rotations in the (i,n) plane which */
/*     eliminate elements in the i-th and n-th planes, respectively. */
/*     q itself is not given, rather the information to recover the */
/*     gv, gw rotations is supplied. */

/*     the subroutine statement is */

/*       subroutine r1mpyq_pair(n,a,lda,b,v,w) */

/*     where */

/*       n is a positive integer input variable set to the number */
/*         of columns of a. */

/*       a is an n by n array. on input a must contain the matrix */
/*         to be postmultiplied by the orthogonal matrix q */
/*         described above. on output a*q has replaced a. */

/*       lda is a positive integer input variable not less than n */
/*         which specifies the leading dimension of the array a. */

/*       b is an input/output array of length n which is multiplied */
/*         by the same orthogonal matrix q. */

/*       v is an input array of length n. v(i) must contain the */
/*         information necessary to recover the givens rotation gv(i) */
/*         described above. */

/*       w is an input array of length n. w(i) must contain the */
/*         information necessary to recover the givens rotation gw(i) */
/*         described above. */

/*     subroutines called */

/*       fortran-supplied ... dabs,dsqrt */

/*     argonne national laboratory. minpack project. march 1980. */
/*     burton s. garbow, kenneth e. hillstrom, jorge j. more */

/*     ********** */
    /* Parameter adjustments */
    --w;
    --v;
    --qtf;
    q_dim1 = ldq;
    q_offset = 1 + q_dim1 * 1;
    q -= q_offset;

    /* Function Body */

/*     apply the first set of givens rotations to a. */

    nm1 = n - 1;
    if (nm1 < 1) {
        return;
    }
    for (nmj = 1; nmj <= nm1; ++nmj) {
	j = n - nmj;
	if (fabs(v[j]) > 1.) {
	    cos = 1 / v[j];
	    sin = sqrt(1 - cos * cos);
	} else {
	    sin = v[j];
	    cos = sqrt(1 - sin * sin);
	}
	for (i = 1; i <= n; ++i) {
	    temp = cos * q[i + j * q_dim1] - sin * q[i + n * q_dim1];
	    q[i + n * q_dim1] = sin * q[i + j * q_dim1] + cos * q[i + n * q_dim1];
	    q[i + j * q_dim1] = temp;
	}
	temp = cos * qtf[j] - sin * qtf[n];
	qtf[n] = sin * qtf[j] + cos * qtf[n];
	qtf[j] = temp;
    }

/*     apply the second set of givens rotations to a. */

    for (j = 1; j <= nm1; ++j) {
	if (fabs(w[j]) > 1.) {
	    cos = 1 / w[j];
	    sin = sqrt(1 - cos * cos);
	} else {
	    sin = w[j];
	    cos = sqrt(1 - sin * sin);
	}
	for (i = 1; i <= n; ++i) {
	    temp = cos * q[i + j * q_dim1] + sin * q[i + n * q_dim1];
	    q[i + n * q_dim1] = -sin * q[i + j * q_dim1] + cos * q[i + n * q_dim1];
	    q[i + j * q_dim1] = temp;
	}
	temp = cos * qtf[j] + sin * qtf[n];
	qtf[n] = -sin * qtf[j] + cos * qtf[n];
	qtf[j] = temp;
    }

/*     last card of subroutine r1mpyq. */

} /* r1mpyq_pair_ */
