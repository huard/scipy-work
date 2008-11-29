""" Unit tests for nonlinear solvers
Author: Ondrej Certik
May 2007
"""

from numpy.testing import *

from scipy.optimize import nonlin
from numpy import matrix, diag
import numpy as np

BROYDEN = [nonlin.broyden1, nonlin.broyden2, nonlin.broyden3]

OTHER = [nonlin.broyden_modified, nonlin.broyden1_modified,
         nonlin.broyden_generalized, nonlin.anderson,
         nonlin.anderson2, nonlin.vackar, nonlin.linearmixing,
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
F.KNOWN_BAD = [nonlin.anderson2]

def F2(x):
    return x
F2.xin = [1,2,3,4,5,6]
F2.KNOWN_BAD = [nonlin.anderson2, nonlin.linearmixing, nonlin.excitingmixing]

def F3(x):
    A = np.mat('-2 1 0; 1 -2 1; 0 1 -2')
    b = np.mat('1 2 3')
    return np.dot(A, x) - b
F3.xin = [1,2,3]
F3.KNOWN_BAD = [nonlin.anderson2]

def F4_powell(x):
    A = 1e4
    return [A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)]
F4_powell.xin = [-1, -2]
F4_powell.KNOWN_BAD = [nonlin.broyden_modified,
                       nonlin.broyden_generalized,
                       nonlin.anderson,
                       nonlin.anderson2,
                       nonlin.linearmixing,
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

class TestNonlinOldTests(TestCase):
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def test_linearmixing(self):
        x = nonlin.linearmixing(F,F.xin,iter=60,alpha=0.5)
        assert nonlin.norm(x)<1e-7
        assert nonlin.norm(F(x))<1e-7

    def test_broyden1(self):
        x= nonlin.broyden1(F,F.xin,iter=11,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_broyden2(self):
        x= nonlin.broyden2(F,F.xin,iter=12,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_broyden3(self):
        x= nonlin.broyden3(F,F.xin,iter=12,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_exciting(self):
        x= nonlin.excitingmixing(F,F.xin,iter=20,alpha=0.5)
        assert nonlin.norm(x)<1e-5
        assert nonlin.norm(F(x))<1e-5

    def test_anderson(self):
        x= nonlin.anderson(F,F.xin,iter=12,alpha=0.03,M=5)
        assert nonlin.norm(x)<0.33

    def test_anderson2(self):
        x= nonlin.anderson2(F,F.xin,iter=12,alpha=0.6,M=5,
                            line_search=False)
        assert nonlin.norm(x)<0.2

    def test_broydengeneralized(self):
        x= nonlin.broyden_generalized(F,F.xin,iter=60,alpha=0.5,M=0,
                                      line_search=True)
        assert nonlin.norm(x)<1e-7
        assert nonlin.norm(F(x))<1e-7
        x= nonlin.broyden_generalized(F,F.xin,iter=61,alpha=0.1,M=1,
                                      line_search=True)
        assert nonlin.norm(x)<2e-4
        assert nonlin.norm(F(x))<2e-4
        x= nonlin.broyden_generalized(F,F.xin,iter=61,alpha=0.1,M=2,
                                      line_search=False)
        assert nonlin.norm(x)<5e-4
        assert nonlin.norm(F(x))<5e-4

    def xtest_broydenmodified(self):
        x= nonlin.broyden_modified(F,F.xin,iter=12,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_broyden1modified(self):
        x= nonlin.broyden1_modified(F,F.xin,iter=35,alpha=1)
        assert nonlin.norm(x)<1e-9
        assert nonlin.norm(F(x))<1e-9

    def test_vackar(self):
        x= nonlin.vackar(F,F.xin,iter=11,alpha=1)
        assert nonlin.norm(x)<1e-8
        assert nonlin.norm(F(x))<1e-8

if __name__ == "__main__":
    run_module_suite()
