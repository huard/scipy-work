"""
Nonlinear solvers
=================

A collection of general-purpose nonlinear multidimensional solvers.
These solvers find x for which F(x)=0. Both x and F can be multidimensional.

They accept the user defined function F, which accepts an ndarray x and it
should return F(x), which should be array-like.

Example:

>>> def F(x):
...    '''Should converge to x=[0,0,0,0,0]'''
...    import numpy
...    d = numpy.array([3,2,1.5,1,0.5])
...    c = 0.01
...    return -d*numpy.array(x)-c*numpy.array(x)**3

>>> from scipy import optimize
>>> x = optimize.broyden2(F,[1,1,1,1,1])

All solvers have the parameter iter (the number of iterations to compute), some
of them have other parameters of the solver, see the particular solver for
details.


Methods
-------

General nonlinear solvers:

.. autosummary::
   :toctree:

   broyden1
   broyden2
   anderson

Simple iterations:

.. autosummary::
   :toctree:

   linearmixing
   vackar
   excitingmixing

"""

import math
import numpy as np
from numpy.linalg import norm, solve
from numpy import asarray, dot, vdot

__all__ = ['broyden1', 'broyden2', 'anderson', 'linearmixing',
           'vackar', 'excitingmixing']

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

class NoConvergence(Exception):
    pass

def maxnorm(x):
    return np.absolute(x).max()

def _as_inexact(x):
    """Return `x` as an array, of either floats or complex floats"""
    x = asarray(x)
    if not isinstance(x.dtype.type, np.inexact):
        return asarray(x, dtype=np.float_)
    return x

def _array_like(x, x0):
    """Return ndarray `x` as same array subclass and shape as `x0`"""
    x.shape = np.shape(x0)
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)

#------------------------------------------------------------------------------
# Generic nonlinear solver machinery
#------------------------------------------------------------------------------

def nonlin_solve(F, x0, jacobian_factory, iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search=True):
    """%s
    Parameters
    ----------
    F : function(x) -> f
        Function whose root to find; should take and return an array-like
        object.
    x0 : array-like
        Initial guess for the solution%s
    iter : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    verbose : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make. If more are needed to
        meet convergence, `NoConvergence` is raised.
    f_tol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    f_rtol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    x_tol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    x_rtol : float, optional
        Relative minimum step size. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : bool, optional
        Whether to make a line search to determine the step size in the
        direction given by the Jacobian approximation. Defaults to ``True``.

    """

    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    func = lambda x: _as_inexact(F(x)).flatten()
    x = _as_inexact(x0).flatten()

    dx = np.inf
    Fx = func(x)
    jacobian = jacobian_factory(x.copy(), Fx)

    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    for n in xrange(maxiter):
        if condition.check(Fx, x, dx):
            break

        if line_search:
            # XXX: determine which of the jacobian approximations stay valid
            #      when the step length is modified

            # XXX: Ensuring descent direction would be useful here?
            dx = -jacobian.solve(Fx)

            # Line search for Wolfe conditions for an objective function
            s = _line_search(func, x, dx)
            step = dx*s
        else:
            dx = -jacobian.solve(Fx)
            step = dx
        x += step
        Fx = func(x)
        jacobian.update(x.copy(), Fx)

        if verbose:
            print "%d:  |F(x)|=%g" % (n, norm(Fx))
    else:
        raise NoConvergence(_array_like(x, x0))

    return _array_like(x, x0)

import minpack2

def _line_search(F, x, dx, c1=1e-4, c2=0.9, maxfev=15, eps=1e-8):
    """
    Perform a line search at `x` to direction `dx` looking for a sufficient
    decrease in the norm of ``F(x + s dx)``.

    The resulting step length will aim toward satisfying the strong Wolfe
    conditions for ``f(s) = ||F(x + s dx/||dx||_2)||_2``, ie.,

    1. f(s)  < f(0) + c1 s f'(0)
    2. |f'(s)| < c2 |f'(0)|

    If no such `s` is found within `maxfev` function evaluations, the
    `s` giving the minimum of `|f(s)|` is returned instead.

    The gradient `f'(s)` is approximated by finite differencing, with
    relative step size of `eps`.

    """
    
    dx_norm = norm(dx)
    dx = dx / dx_norm
    if dx_norm == 0:
        raise ValueError('Invalid search direction')

    def func(s):
        return norm(F(x + s*dx))

    def grad(s, f0):
        ds = (abs(s) + norm(x)) * eps
        return (func(s + ds) - f0) / ds

    xtol = 1e-2
    stpmin = 1e-2 * dx_norm
    stpmax = 50. * dx_norm
    stp = dx_norm

    f = func(0.)
    g = grad(0., f)

    isave = np.zeros(2, dtype=np.intc)
    dsave = np.zeros(13, dtype=np.float_)
    task = 'START'

    best_stp = stp
    best_stp_f = f

    for k in xrange(maxfev//2):
        stp, f, g, task = minpack2.dcsrch(stp, f, g, c1, c2, xtol, task,
                                          stpmin, stpmax, isave, dsave)
        if task[:2] == 'FG':
            f = func(stp)
            g = grad(stp, f)
        else:
            break
        if f < best_stp_f:
            best_stp = stp
            best_stp_f = f
    else:
        stp = best_stp
        task = 'WARNING'

    if task[:5] == 'ERROR':
        stp = best_stp

    stp /= dx_norm
    return stp

class Jacobian(object):
    def __init__(self, x0, f0, **kw):
        raise NotImplementedError
    def solve(self, rhs):
        raise NotImplementedError
    def update(self, x, F):
        raise NotImplementedError

class TerminationCondition(object):
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """
    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 iter=None, norm=maxnorm):

        if f_tol is None:
            f_tol = np.finfo(np.float_).eps ** (1./3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf
        
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol

        self.norm = maxnorm
        self.iter = iter

        self.f0_norm = None
        self.iteration = 0
        
    def check(self, f, x, dx):
        self.iteration += 1
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)

        if self.f0_norm is None:
            self.f0_norm = f_norm
            
        if f_norm == 0:
            return True

        if self.iter is not None:
            # backwards compatibility with Scipy 0.6.0
            return self.iteration > self.iter

        # NB: condition must succeed for rtol=inf even if norm == 0
        return ((f_norm <= self.f_tol and f_norm/self.f_rtol <= self.f0_norm)
                and (dx_norm <= self.x_tol and dx_norm/self.x_rtol <= x_norm))

#------------------------------------------------------------------------------
# Full Broyden / Quasi-Newton variants
#------------------------------------------------------------------------------

class GenericBroyden(Jacobian):
    def __init__(self, x0, f0):
        self.last_f = f0
        self.last_x = x0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        raise NotImplementedError

    def update(self, x, f):
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        self.last_f = f
        self.last_x = x

class BroydenFirst(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \"Broyden's good method\".
    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).""")
    
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.Jm = (-1./alpha) * np.identity(x0.size)

    def solve(self, f):
        return solve(self.Jm, f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.Jm += (df - dot(self.Jm, dx))[:,None] * dx[None,:]/dx_norm**2

class BroydenSecond(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".
    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).""")
    
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.Gm = -alpha * np.identity(x0.size)

    def solve(self, f):
        return dot(self.Gm, f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.Gm += (dx - dot(self.Gm, df))[:,None] * df[None,:]/df_norm**2

#------------------------------------------------------------------------------
# Broyden-like (restricted memory)
#------------------------------------------------------------------------------

class Anderson(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using (extended) Anderson mixing.

    The jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversion and MxN multiplication is required. [Ey]_

    .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).

    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter for numerical stability.
        Compared to unity, good values of the order of 0.01.""")

    def __init__(self, x0, f0, alpha=0.1, w0=0.01, M=5):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f):
        dx = -self.alpha*f
        for m in xrange(len(self.dx)):
            dx += self.gamma[m]*(self.dx[m] + self.alpha*self.df[m])
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if self.M == 0:
            return
        
        self.dx.append(dx)
        self.df.append(df)

        while len(self.dx) > self.M:
            self.dx.pop(0)
            self.df.pop(0)

        n = len(self.dx)
        a = np.zeros((n, n))
        
        for i in xrange(n):
            for j in xrange(i, n):
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1+wd)*vdot(self.df[i], self.df[j])

        a += np.triu(a, 1).T.conj()

        dFF = np.empty(n)
        for k in xrange(n):
            dFF[k] = vdot(self.df[k], f)

        self.gamma = solve(a, dFF)

#------------------------------------------------------------------------------
# Simple iterations
#------------------------------------------------------------------------------

class Vackar(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using diagonal Broyden Jacobian approximation.
    
    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).""")

    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.d = np.ones_like(x0)/alpha

    def solve(self, f):
        return -f / self.d

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d*dx)*dx/dx_norm**2

class LinearMixing(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).""")

    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha

    def solve(self, f):
        return -f*self.alpha

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass

class ExcitingMixing(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.""")

    def __init__(self, x0, f0, alpha=0.1, alphamax=1.0):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = alpha*np.ones_like(x0)

    def solve(self, f):
        return -f*self.beta

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        incr = f*self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)

#------------------------------------------------------------------------------
# Wrapper functions
#------------------------------------------------------------------------------

def _broyden_wrapper(name, jac):
    import inspect
    args, varargs, varkw, defaults = inspect.getargspec(jac.__init__)
    kwargs = dict(zip(args[-len(defaults):], defaults))
    kw_str = ", ".join(["%s=%r" % (k, v) for k, v in kwargs.items()])
    if kw_str:
        kw_str = ", " + kw_str
    kwkw_str = ", ".join(["%s=%s" % (k, k) for k, v in kwargs.items()])
    if kwkw_str:
        kwkw_str = ", " + kwkw_str

    wrapper = ("def %s(F, xin, iter=None %s, verbose=False, maxiter=None, "
               "f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, "
               "tol_norm=None, line_search=True):\n"
               "    jac = lambda x, f: %s(x, f %s)\n"
               "    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,\n"
               "        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search)")
    wrapper = wrapper % (name, kw_str, jac.__name__, kwkw_str)
    ns = {}
    ns.update(globals())
    exec wrapper in ns
    func = ns[name]
    func.__doc__ = jac.__doc__
    return func

broyden1 = _broyden_wrapper('broyden1', BroydenFirst)
broyden2 = _broyden_wrapper('broyden2', BroydenSecond)

anderson = _broyden_wrapper('anderson', Anderson)

linearmixing = _broyden_wrapper('linearmixing', LinearMixing)
vackar = _broyden_wrapper('vackar', Vackar)
excitingmixing = _broyden_wrapper('excitingmixing', ExcitingMixing)
