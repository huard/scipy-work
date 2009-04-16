/*
 * Evaluate orthogonal polynomials.
 *
 * See Section 22.18 in Abramowitz & Stegun.
 *
 *
 * Copyright (C) 2009 Pauli Virtanen
 * Distributed under the same license as Scipy.
 *
 */

#include <stdlib.h>

double eval_poly_jacobi(int k, double alpha, double beta, double x)
{
    double a, b, c, d, f;
    int m, n;

    n = k;
    a = 1;
    d = 1;
    f = 1-x;
    for (m = n; m > 0; --m) {
        b = (n - m + 1)*(alpha + beta + n + m);
        c = 2*m*(alpha+m);
        a = 1 - b*f/c*a;
        d *= (alpha+m)/m;
    }
    return a*d;
}

double eval_poly_gegenbauer(int k, double alpha, double x)
{
    double a, b, c, d, f;
    int n, m;

    if (k % 2 == 0) {
        n = k/2;
        a = 1;
        d = 1;
        f = x*x;
        for (m = n; m > 0; --m) {
            b = 2*(n-m+1)*(alpha+n+m-1);
            c = m*(2*m-1);
            a = 1 - b*f/c*a;
            d *= -(alpha + m-1)/m;
        }
    } else {
        n = (k-1)/2;
        a = 1;
        d = 2*x*alpha;
        f = x*x;
        for (m = n; m > 0; --m) {
            b = 2*(n-m+1)*(alpha+n+m);
            c = m*(2*m+1);
            a = 1 - b*f/c*a;
            d *= -(alpha + m)/m;
        }
    }
    return a*d;
}

double eval_poly_chebyt(int k, double x)
{
    double a, b, c, d, f;
    int n, m;

    if (k % 2 == 0) {
        n = k/2;
        a = 1;
        d = 1;
        f = x*x;
        for (m = n; m > 0; --m) {
            b = 2*(n-m+1)*(n+m-1);
            c = m*(2*m-1);
            a = 1 - b*f/c*a;
            d = -d;
        }
    } else {
        n = (k-1)/2;
        a = 1;
        d = k*2*x;
        f = x*x;
        for (m = n; m > 0; --m) {
            b = 2*(n-m+1)*(n+m);
            c = m*(2*m+1);
            a = 1 - b*f/c*a;
            d = -d;
        }
    }
    return a*d;
}

double eval_poly_chebyu()
{
}

double eval_poly_chebys()
{
}

double eval_poly_sh_chebyt()
{
}

double eval_poly_sh_chebyu()
{
}

double eval_poly_legendre()
{
}

double eval_poly_sh_legendre()
{
}

double eval_poly_genlaguerre()
{
}

double eval_poly_laguerre()
{
}

double eval_poly_hermite()
{
}

double eval_poly_hermite2()
{
}
