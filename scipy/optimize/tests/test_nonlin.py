""" Unit tests for nonlinear solvers
Author: Ondrej Certik
May 2007
"""

from numpy.testing import *

from scipy.optimize import nonlin
from numpy import matrix, diag, dot
from numpy.linalg import inv
import numpy as np

BROYDEN = [nonlin.broyden1, nonlin.broyden2]
OTHER = [nonlin.anderson, nonlin.vackar, nonlin.linearmixing,
         nonlin.excitingmixing]

def F(x):
    def p3(y):
        return float(y.T*y)*y
    x = np.asmatrix(x).T
    d=matrix(diag([3,2,1.5,1,0.5]))
    c=0.01
    f=-d*x-c*p3(x)
    return f
F.xin = [1,1,1,1,1]
F.KNOWN_BAD = []

def F2(x):
    return x
F2.xin = [1,2,3,4,5,6]
F2.KNOWN_BAD = [nonlin.linearmixing, nonlin.excitingmixing]

def F3(x):
    A = np.mat('-2 1 0; 1 -2 1; 0 1 -2')
    b = np.mat('1 2 3')
    return np.dot(A, x) - b
F3.xin = [1,2,3]
F3.KNOWN_BAD = []

def F4_powell(x):
    A = 1e4
    return [A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)]
F4_powell.xin = [-1, -2]
F4_powell.KNOWN_BAD = [nonlin.anderson, nonlin.linearmixing,
                       nonlin.excitingmixing]

class TestNonlin(object):
    """
    Check the Broyden methods for a few test problems.

    broyden1, broyden2, and broyden3 must succeed for all functions.
    Some of the others don't -- tests in KNOWN_BAD are skipped.

    """

    def _check_func(self, f, func, f_tol=1e-2):
        x = func(f, f.xin, f_tol=f_tol, maxiter=100)
        assert np.absolute(f(x)).max() < f_tol

    @dec.knownfailureif(True)
    def _check_func_fail(self, *a, **kw):
        pass

    def test_problem(self):
        for f in [F, F2, F3, F4_powell]:
            for func in BROYDEN + OTHER:
                if func in f.KNOWN_BAD and func not in BROYDEN:
                    #yield self._check_func_fail, f, func
                    continue
                yield self._check_func, f, func


class TestSecant(TestCase):
    """Check that some Jacobian approximations satisfy the secant condition"""

    xs = [np.array([1,2,3,4,5]),
          np.array([2,3,4,5,1]),
          np.array([3,4,5,1,2]),
          np.array([4,5,1,2,3]),
          np.array([5,1,2,3,6]),]
    fs = [x**2 - 1 for x in xs]

    def _check_secant(self, jac_cls, npoints=1, **kw):
        """
        Check that the given Jacobian approximation satisfies secant
        conditions for last `npoints` points.
        """
        jac = jac_cls(self.xs[0], self.fs[0], **kw)
        for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            jac.update(x, f)

            for k in xrange(min(npoints, j+1)):
                dx = self.xs[j-k+1] - self.xs[j-k]
                df = self.fs[j-k+1] - self.fs[j-k]
                assert np.allclose(dx, jac.solve(df))

            # Check that the `npoints` secant bound is strict
            if j >= npoints:
                dx = self.xs[j-npoints+1] - self.xs[j-npoints]
                df = self.fs[j-npoints+1] - self.fs[j-npoints]
                assert not np.allclose(dx, jac.solve(df))

    def test_broyden1(self):
        self._check_secant(nonlin.BroydenFirst)
        
    def test_broyden2(self):
        self._check_secant(nonlin.BroydenSecond)

    def test_broyden1_sherman_morrison(self):
        # Check that BroydenFirst is as expected for the 1st iteration
        jac = nonlin.BroydenFirst(self.xs[0], self.fs[0], alpha=0.1)
        jac.update(self.xs[1], self.fs[1])

        df = self.fs[1] - self.fs[0]
        dx = self.xs[1] - self.xs[0]
        j0 = -1./0.1 * np.eye(5)
        j0 += (df - dot(j0, dx))[:,None] * dx[None,:] / dot(dx, dx)

        assert np.allclose(inv(j0), jac.Gm)

    def test_anderson(self):
        # Anderson mixing (with w0=0) satisfies secant conditions
        # for the last M iterates, see [Ey]_
        #
        # .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).
        self._check_secant(nonlin.Anderson, M=3, w0=0, npoints=3)

class TestNonlinOldTests(TestCase):
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def test_broyden1(self):
        x= nonlin.broyden1(F,F.xin,iter=11,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_broyden2(self):
        x= nonlin.broyden2(F,F.xin,iter=12,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_anderson(self):
        x= nonlin.anderson(F,F.xin,iter=12,alpha=0.03,M=5)
        assert nonlin.norm(x)<0.33

    def test_linearmixing(self):
        x = nonlin.linearmixing(F,F.xin,iter=60,alpha=0.5)
        assert nonlin.norm(x)<1e-7
        assert nonlin.norm(F(x))<1e-7

    def test_exciting(self):
        x= nonlin.excitingmixing(F,F.xin,iter=20,alpha=0.5)
        assert nonlin.norm(x)<1e-5
        assert nonlin.norm(F(x))<1e-5

    def test_vackar(self):
        x= nonlin.vackar(F,F.xin,iter=11,alpha=1)
        assert nonlin.norm(x)<1e-8
        assert nonlin.norm(F(x))<1e-8

if __name__ == "__main__":
    run_module_suite()
