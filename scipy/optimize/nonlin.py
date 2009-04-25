r"""
Nonlinear solvers
=================

.. currentmodule:: scipy.optimize

This is a collection of general-purpose nonlinear multidimensional
solvers.  These solvers find *x* for which :math:`F(x)=0`. Both *x*
and *F* can be multidimensional.

Example:

>>> def F(x):
...    '''Should converge to x=[0,0,0,0,0]'''
...    d = [3,2,1.5,1,0.5]
...    c = 0.01
...    return -d * x - c*x**3
>>> import scipy.optimize
>>> x = scipy.optimize.broyden2(F, [1,1,1,1,1])


Routines
--------

Large-scale nonlinear solvers:

.. autosummary::

   newton_krylov
   anderson

General nonlinear solvers:

.. autosummary::

   broyden1
   broyden2

Simple iterations:

.. autosummary::

   excitingmixing
   linearmixing
   vackar


Example: large problem
----------------------

Suppose that we needed to solve the following integrodifferential
equation on the square :math:`[0,1]\times[0,1]`:

.. math::

   \nabla^2 P = 10 \left(\int_0^1\int_0^1\cosh(P)\,dx\,dy\right)^2

with :math:`P(x,1) = 1` and :math:`P=0` elsewhere on the boundary of
the square.

The solution can be found using the `newton_krylov` solver:

.. plot::

   import numpy as np
   from scipy.optimize import newton_krylov
   from numpy import cosh, zeros_like, mgrid, zeros

   # parameters
   nx, ny = 75, 75
   hx, hy = 1./(nx-1), 1./(ny-1)

   P_left, P_right = 0, 0
   P_top, P_bottom = 1, 0

   def residual(P):
       d2x = zeros_like(P)
       d2y = zeros_like(P)

       d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
       d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
       d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

       d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
       d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
       d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

       return d2x + d2y - 10*cosh(P).mean()**2

   # solve
   guess = zeros((nx, ny), float)
   sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
   print 'Residual', abs(residual(sol)).max()

   # visualize
   import matplotlib.pyplot as plt
   x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
   plt.pcolor(x, y, sol)
   plt.colorbar()
   plt.show()

"""
# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as Scipy.

import sys
import numpy as np
from numpy.linalg import norm, solve
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import minpack2

__all__ = [
    'broyden1', 'broyden2', 'anderson', 'linearmixing',
    'vackar', 'excitingmixing', 'newton_krylov',
    # Deprecated functions:
    'broyden_generalized', 'broyden1_modified', 'broyden_modified',
    'anderson2', 'broyden3',]

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
    x = np.reshape(x, np.shape(x0))
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)

#------------------------------------------------------------------------------
# Generic nonlinear solver machinery
#------------------------------------------------------------------------------

_doc_parts = dict(
    params_basic="""
    F : function(x) -> f
        Function whose root to find; should take and return an array-like
        object.
    x0 : array-like
        Initial guess for the solution
    """.strip(),
    params_extra="""
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
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual.

    Returns
    -------
    sol : array-like
        An array (of similar array type as `x0`) containing the final solution.

    Raises
    ------
    NoConvergence
        When a solution was not found.

    """.strip()
)

def _set_doc(obj):
    if obj.__doc__:
        obj.__doc__ = obj.__doc__ % _doc_parts

def nonlin_solve(F, x0, jacobian_factory, iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search=True,
                 callback=None):
    """
    Find a root of a function, using given Jacobian approximation.

    Parameters
    ----------
    %(params_basic)s
    jacobian_factory : function(x0, f0, func)
        Callable that constructs a Jacobian approximation. It is passed
        the initial guess `x0`, the initial residual `f0` and a function
        object `func` that evaluates the residual.
    %(params_extra)s
    """

    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    func = lambda z: _as_inexact(F(_array_like(z, x0))).flatten()
    x = _as_inexact(x0).flatten()

    dx = np.inf
    Fx = func(x)
    jacobian = jacobian_factory(x.copy(), Fx, func)

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

            if abs(dx).max() == 0:
                # Jacobian was faulty, fall back to gradient direction
                dx = -Fx

            # Line search for Wolfe conditions for an objective function
            s = _line_search(func, x, dx)
            step = dx*s
        else:
            dx = -jacobian.solve(Fx)
            step = dx
        x += step
        Fx = func(x)
        jacobian.update(x.copy(), Fx)

        if callback:
            callback(x, Fx)

        if verbose:
            print "%d:  |F(x)| = %g; step %g" % (n, norm(Fx), s)
            sys.stdout.flush()
    else:
        raise NoConvergence(_array_like(x, x0))

    return _array_like(x, x0)

_set_doc(nonlin_solve)

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
    x_norm = norm(x)
    dx = dx / dx_norm
    if dx_norm == 0:
        raise ValueError('Invalid search direction')

    def func(s):
        return norm(F(x + s*dx))

    def grad(s, f0):
        ds = (abs(s) + x_norm) * eps
        return (func(s + ds) - f0) / ds

    xtol = 1e-2
    stpmin = 1e-4 * dx_norm
    stpmax = 50. * dx_norm
    stp = dx_norm

    f = func(0.)
    g = grad(0., f)

    if g > 0:
        # The direction given is not a descent direction; go the other way.
        g = -g
        dx = -dx
        sign = -1
    else:
        sign = 1

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
    return stp*sign

class Jacobian(object):
    def __init__(self, x0, f0, func, **kw):
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
    def __init__(self, x0, f0, func):
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
    """
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \"Broyden's good method\".

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0, func)
        self.Gm = -alpha * np.identity(x0.size)

    def solve(self, f):
        return dot(self.Gm, f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        s = dot(self.Gm.T, dx)
        y = dot(self.Gm, df)
        self.Gm += (dx - y)[:,None] * s[None,:] / dot(dx, y)
        
class BroydenSecond(GenericBroyden):
    """
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0, func)
        self.Gm = -alpha * np.identity(x0.size)

    def solve(self, f):
        return dot(self.Gm, f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.Gm += (dx - dot(self.Gm, df))[:,None] * df[None,:]/df_norm**2
        
#------------------------------------------------------------------------------
# Broyden-like (restricted memory)
#------------------------------------------------------------------------------

class Anderson(GenericBroyden):
    """
    Find a root of a function, using (extended) Anderson mixing.

    The jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversion and MxN multiplication is required. [Ey]_

    .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter for numerical stability.
        Compared to unity, good values of the order of 0.01.
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1, w0=0.01, M=5):
        GenericBroyden.__init__(self, x0, f0, func)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f):
        dx = -self.alpha*f

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n)
        for k in xrange(n):
            df_f[k] = vdot(self.df[k], f)
        gamma = solve(self.a, df_f)

        for m in xrange(len(self.dx)):
            dx += gamma[m]*(self.dx[m] + self.alpha*self.df[m])
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
        self.a = a

#------------------------------------------------------------------------------
# Simple iterations
#------------------------------------------------------------------------------

class Vackar(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.
    
    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0, func)
        self.d = np.ones_like(x0)/alpha

    def solve(self, f):
        return -f / self.d

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d*dx)*dx/dx_norm**2

class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0, func)
        self.alpha = alpha

    def solve(self, f):
        return -f*self.alpha

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass

class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, x0, f0, func, alpha=0.1, alphamax=1.0):
        GenericBroyden.__init__(self, x0, f0, func)
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
# Iterative/Krylov approximated Jacobians
#------------------------------------------------------------------------------

class KrylovJacobian(Jacobian):
    r"""
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : {'lgmres', 'gmres', 'bicgstab', 'cgs', 'minres'} or function
        Krylov method to use to approximate the Jacobian.
        Can be a string, or a function implementing the same interface as
        the iterative solvers in `scipy.sparse.linalg`.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_tol, inner_maxiter, inner_M, ...
        Parameters to pass on to the \"inner\" Krylov solver.
        See `scipy.sparse.linalg.gmres` for details.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    %(params_extra)s

    See Also
    --------
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by numerical
    differentiation:

    .. math:: J v \approx (f(x + \omega*v/|v|) - f(x)) / \omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    Scipy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [KK]_,
    and for the LGMRES sparse inverse method, see [BJM]_.

    References
    ----------
    .. [KK] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2003).
    .. [BJM] A.H. Baker and E.R. Jessup and T. Manteuffel,
             SIAM J. Matrix Anal. Appl. 26, 962 (2005).

    """

    def __init__(self, x0, f0, func, rdiff=None,
                 method='lgmres',
                 inner_tol=1e-6, inner_maxiter=20, inner_M=None,
                 outer_k=6, **kw):

        self.x0 = x0
        self.f0 = f0
        self.func = func

        if rdiff is None:
            rdiff = np.finfo(x0.dtype).eps ** (1./2)

        self.rdiff = rdiff
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            ).get(method, method)

        self.method_kw = dict(tol=inner_tol, maxiter=inner_maxiter,
                              M=inner_M)

        if self.method is scipy.sparse.linalg.gmres:
            # Replace GMRES's outer iteration with Newton steps
            self.method_kw['restrt'] = inner_maxiter
            self.method_kw['maxiter'] = 1
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            # Replace LGMRES's outer iteration with Newton steps
            self.method_kw['maxiter'] = 1
            # Carry LGMRES's `outer_v` vectors across nonlinear iterations
            self.method_kw.setdefault('outer_v', [])
            # But don't carry the corresponding Jacobian*v products, in case
            # the Jacobian changes a lot in the nonlinear step
            #
            # XXX: some trust-region inspired ideas might be more efficient...
            self.method_kw.setdefault('store_outer_Av', False)

        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError("Unknown parameter %s" % key)
            self.method_kw[key[6:]] = value

        self.op = scipy.sparse.linalg.LinearOperator(
            shape=(f0.size, x0.size), matvec=self._mul, dtype=self.f0.dtype)
        self._update_diff_step()

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def _mul(self, v):
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs):
        sol, info = self.method(self.op, rhs, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()


#------------------------------------------------------------------------------
# Wrapper functions
#------------------------------------------------------------------------------

def _nonlin_wrapper(name, jac):
    """
    Construct a solver wrapper with given name and jacobian approx.

    It inspects the keyword arguments of ``jac.__init__``, and allows to
    use the same arguments in the wrapper function, in addition to the
    keyword arguments of `nonlin_solve`

    """
    import inspect
    args, varargs, varkw, defaults = inspect.getargspec(jac.__init__)
    kwargs = dict(zip(args[-len(defaults):], defaults))
    kw_str = ", ".join(["%s=%r" % (k, v) for k, v in kwargs.items()])
    if kw_str:
        kw_str = ", " + kw_str
    kwkw_str = ", ".join(["%s=%s" % (k, k) for k, v in kwargs.items()])
    if kwkw_str:
        kwkw_str = ", " + kwkw_str

    # Construct the wrapper function so that it's keyword arguments
    # are visible in pydoc.help etc.
    wrapper = """
def %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None, 
             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, 
             tol_norm=None, line_search=True, **kw):
    jac = lambda x, f, func: %(jac)s(x, f, func %(kwkw)s, **kw)
    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search)
"""
    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__,
                             kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    exec wrapper in ns
    func = ns[name]
    func.__doc__ = jac.__doc__
    _set_doc(func)
    return func

broyden1 = _nonlin_wrapper('broyden1', BroydenFirst)
broyden2 = _nonlin_wrapper('broyden2', BroydenSecond)
anderson = _nonlin_wrapper('anderson', Anderson)
linearmixing = _nonlin_wrapper('linearmixing', LinearMixing)
vackar = _nonlin_wrapper('vackar', Vackar)
excitingmixing = _nonlin_wrapper('excitingmixing', ExcitingMixing)
newton_krylov = _nonlin_wrapper('newton_krylov', KrylovJacobian)


# Deprecated functions

@np.deprecate
def broyden_generalized(*a, **kw):
    """Use anderson(..., w0=0) instead"""
    kw.setdefault('w0', 0)
    return anderson(*a, **kw)

@np.deprecate
def broyden1_modified(*a, **kw):
    """Use broyden1 instead"""
    return broyden1(*a, **kw)

@np.deprecate
def broyden_modified(*a, **kw):
    """Use anderson instead"""
    return anderson(*a, **kw)

@np.deprecate
def anderson2(*a, **kw):
    """anderson2 was faulty; use anderson instead"""
    return anderson(*a, **kw)

@np.deprecate
def broyden3(*a, **kw):
    """Use broyden2 instead"""
    return broyden2(*a, **kw)
