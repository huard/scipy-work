r"""
Nonlinear solvers
=================

.. currentmodule:: scipy.optimize

This is a collection of general-purpose nonlinear multidimensional
solvers.  These solvers find *x* for which :math:`F(x)=0`. Both *x*
and *F* can be multidimensional.

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

 
References
----------

.. [BS] P. N. Brown and Y. Saad. ''Globally convergent techniques in
   nonlinear Newton-Krylov algorithms''

   ftp://ftp.cs.umn.edu/dept/users/saad/reports/PDF/RIACS-89-57.pdf

.. 


Examples
========

Small problem
-------------

>>> def F(x):
...    '''Should converge to x=[0,0,0,0,0]'''
...    d = [3,2,1.5,1,0.5]
...    c = 0.01
...    return -d * x - c*x**3
>>> import scipy.optimize
>>> x = scipy.optimize.broyden2(F, [1,1,1,1,1])

Large problem
-------------

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
from scipy.linalg import norm, solve, inv, qr, svd, lstsq
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
import scipy.lib.blas as blas
import inspect
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

def nonlin_solve(F, x0, jacobian='krylov', iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search='armijo', callback=None):
    """
    Find a root of a function, using given Jacobian approximation.

    Parameters
    ----------
    %(params_basic)s
    jacobian : Jacobian
        A Jacobian approximation.
    %(params_extra)s

    References
    ----------
    .. [KIM] C. T. Kelley, \"Iterative Methods for Linear and Nonlinear
       Equations\". Society for Industrial and Applied Mathematics. (1995)
       http://www.siam.org/books/kelley/

    """

    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    func = lambda z: _as_inexact(F(_array_like(z, x0))).flatten()
    x = _as_inexact(x0).flatten()

    dx = np.inf
    Fx = func(x)
    Fx_norm = norm(Fx)

    jacobian = asjacobian(jacobian)
    jacobian.setup(x.copy(), Fx, func)

    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    if line_search not in (None, 'armijo', 'wolfe'):
        raise ValueError("Invalid line search")

    # Solver tolerance selection
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3

    for n in xrange(maxiter):
        if condition.check(Fx, x, dx):
            break

        # The tolerance, as computed for scipy.sparse.linalg.* routines
        tol = min(eta, eta*Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        if norm(dx) == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        if line_search == 'wolfe':
            # Line search for Wolfe conditions for an objective function
            s, x, Fx, Fx_norm_new = line_search_wolfe(func, x, dx, Fx, Fx_norm)
        elif line_search == 'armijo':
            s, x, Fx, Fx_norm_new = line_search_armijo(func, x, dx, Fx, Fx_norm)

        if line_search is None or s == 0.0:
            # If no line search, or if line search failed, take the full step
            # and hope for the best.
            s = 1.0
            x += dx
            Fx = func(x)
            Fx_norm_new = norm(Fx)

        jacobian.update(x.copy(), Fx)

        if callback:
            callback(x, Fx)

        # Adjust forcing parameters for inexact methods
        eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma*eta**2))

        Fx_norm = Fx_norm_new

        # Print status
        if verbose:
            sys.stdout.write("%d:  |F(x)| = %g; step %g; tol %g\n" % (
                n, norm(Fx), s, eta))
            sys.stdout.flush()
    else:
        raise NoConvergence(_array_like(x, x0))

    return _array_like(x, x0)

_set_doc(nonlin_solve)


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
# Line searches
#------------------------------------------------------------------------------

def line_search_armijo(F, x, dx, Fx=None, Fx_norm=None, alpha=1e-4):
    """
    Perform a line search at `x` to direction `dx` looking for a sufficient
    decrease in the norm of ``F(x + s dx)``.

    The resulting step length will aim toward satisfying the Armijo condition

        || F(x + s dx) ||_2 < (1 - alpha s) || F(x + s dx) ||_2

    Parameters
    ----------
    F
        The function
    x
        Position
    dx
        Search direction
    Fx : optional
        F(x)
    alpha : float, optional
        Armijo parameter

    Returns
    -------
    s
        Selected step length
    x_new
        x + s dx
    Fx_new
        F(x + s dx)
    Fx_norm_new
        Norm of Fx_new

    """

    # XXX: polynomial model?

    if Fx is None:
        Fx = F(x)
    if Fx_norm is None:
        Fx_norm = norm(Fx)

    x0 = x
    Fx0 = Fx
    Fx_norm0 = Fx_norm

    s = 1.0
    for k in xrange(6):
        x_new = x + s * dx
        Fx_new = F(x_new)
        Fx_norm_new = norm(Fx_new)

        if Fx_norm_new < (1 - alpha * s) * Fx_norm:
            break

        s /= 2
    else:
        # Give up; it's probably not a descent direction.
        # Could fail here, but just return the shortest step
        s, x_new, Fx_new, Fx_norm_new = 0., x0, Fx0, Fx_norm0

    return s, x_new, Fx_new, Fx_norm_new

def _safe_norm(v):
    if np.isinf(v).any():
        return np.inf
    return norm(v)

def line_search_wolfe(F, x0, dx, Fx=None, Fx_norm=None,
                      c1=1e-4, c2=0.9, maxfev=15, eps=1e-8):
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

    x = x0
    x0_norm = norm(x)
    dx_norm = norm(dx)
    dx = dx / dx_norm

    if dx_norm == 0:
        raise ValueError('Invalid search direction')

    def grad(s, f0):
        ds = (abs(s) + x0_norm) * eps
        return (_safe_norm(F(x0 + (s+ds)*dx)) - f0) / ds

    xtol = 1e-2
    stpmin = 1e-2 * dx_norm
    stpmax = 50. * dx_norm
    stp = dx_norm

    if Fx is None:
        Fx = F(x)
    if Fx_norm is None:
        Fx_norm = _safe_norm(Fx)

    f = Fx_norm
    g = grad(0., f)

    isave = np.zeros(2, dtype=np.intc)
    dsave = np.zeros(13, dtype=np.float_)
    task = 'START'

    best_stp = (0., x, Fx, Fx_norm)
    last_stp = 0.

    for k in xrange(maxfev//2):
        stp, f2, g2, task = minpack2.dcsrch(stp, f, g, c1, c2, xtol, task,
                                            stpmin, stpmax, isave, dsave)

        if stp != last_stp:
            x = x0 + stp*dx
            Fx = F(x)
            f = Fx_norm = _safe_norm(Fx)
            g = grad(stp, f)
            last_stp = stp

        if Fx_norm < best_stp[-1]:
            best_stp = (stp, x, Fx, Fx_norm)

        if task[:2] != 'FG':
            break
    else:
        task = 'ITER'

    if task[:5] == 'ERROR' or task == 'ITER':
        stp, x, Fx, Fx_norm = best_stp

    stp /= dx_norm
    return stp, x, Fx, Fx_norm


#------------------------------------------------------------------------------
# Generic Jacobian approximation
#------------------------------------------------------------------------------

class Jacobian(object):
    """
    Common interface for Jacobians or Jacobian approximations.

    The optional methods come useful when implementing trust region
    etc.  algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.
        
    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    def __init__(self, **kw):
        names = ["solve", "update", "matvec", "rmatvec", "rsolve",
                 "matmat", "todense", "shape", "dtype"]
        for name, value in kw.items():
            if name not in names:
                raise ValueError("Unknown keyword argument %s" % name)
            if value is not None:
                setattr(self, name, kw[name])

        if hasattr(self, 'todense'):
            self.__array__ = lambda: self.todense()

    def aspreconditioner(self):
        return InverseJacobian(self)

    def solve(self, v, tol=0):
        raise NotImplementedError

    def update(self, x, F):
        pass

    def setup(self, x, F, func):
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        if self.__class__.setup is Jacobian.setup:
            # Call on the first point unless overridden
            self.update(self, x, F)

class InverseJacobian(object):
    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.matvec = jacobian.solve
        self.update = jacobian.update
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    @property
    def shape(self):
        return self.jacobian.shape

    @property
    def dtype(self):
        return self.jacobian.dtype

def asjacobian(J):
    """
    Convert given object to one suitable for use as a Jacobian.
    """
    spsolve = scipy.sparse.linalg.spsolve
    if isinstance(J, Jacobian):
        return J
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    elif isinstance(J, np.ndarray):
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        J = np.atleast_2d(np.asarray(J))
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')

        return Jacobian(matvec=lambda v: dot(J, v),
                        rmatvec=lambda v: dot(J.conj().T, v),
                        solve=lambda v: solve(J, v),
                        rsolve=lambda v: solve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif scipy.sparse.isspmatrix(J):
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        return Jacobian(matvec=lambda v: J*v,
                        rmatvec=lambda v: J.conj().T * v,
                        solve=lambda v: spsolve(J, v),
                        rsolve=lambda v: spsolve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        return Jacobian(matvec=getattr(J, 'matvec'),
                        rmatvec=getattr(J, 'rmatvec'),
                        solve=J.solve,
                        rsolve=getattr(J, 'rsolve'),
                        update=getattr(J, 'update'),
                        setup=getattr(J, 'setup'),
                        dtype=J.dtype,
                        shape=J.shape)
    elif callable(J):
        # Assume it's a function J(x) that returns the Jacobian
        class Jac(Jacobian):
            def update(self, x, F):
                self.x = x
            def solve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.isspmatrix(m):
                    return spsolve(m, v)
                else:
                    raise ValueError("Unknown matrix type")
            def matvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.isspmatrix(m):
                    return m*v
                else:
                    raise ValueError("Unknown matrix type")
            def rsolve(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.isspmatrix(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError("Unknown matrix type")
            def rmatvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.isspmatrix(m):
                    return m.conj().T * v
                else:
                    raise ValueError("Unknown matrix type")
        return Jac()
    elif isinstance(J, str):
        return dict(broyden1=BroydenFirst,
                    broyden2=BroydenSecond,
                    anderson=Anderson,
                    vackar=Vackar,
                    linearmixing=LinearMixing,
                    excitingmixing=ExcitingMixing,
                    krylov=KrylovJacobian)[J]()
    else:
        raise TypeError('Cannot convert object to a Jacobian')


#------------------------------------------------------------------------------
# Broyden
#------------------------------------------------------------------------------

class GenericBroyden(Jacobian):
    def setup(self, x0, f0, func):
        Jacobian.setup(self, x0, f0, func)
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

class LowRankMatrix(object):
    r"""
    A matrix represented as

    .. math:: \alpha I + \sum_{n=0}^{n=M} c_n d_n^\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """

    def __init__(self, alpha, n, dtype):
        self.alpha = alpha
        self.cs = []
        self.ds = []
        self.n = n
        self.dtype = dtype
        self.collapsed = None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        axpy, scal, dotc = blas.get_blas_funcs(['axpy', 'scal', 'dotc'],
                                               cs[:1] + [v])
        w = alpha * v
        for c, d in zip(cs, ds):
            a = dotc(d, v)
            w = axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        """Evaluate w = M^-1 v"""
        if len(cs) == 0:
            return v/alpha

        # (B + C D^H)^-1 = B^-1 - B^-1 C (I + D^H B^-1 C)^-1 D^H B^-1

        axpy, dotc = blas.get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])

        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)
        for i, d in enumerate(ds):
            for j, c in enumerate(cs):
                A[i,j] += dotc(d, c)

        q = np.zeros(len(cs), dtype=c0.dtype)
        for j, d in enumerate(ds):
            q[j] = dotc(d, v)
        q /= alpha
        q = solve(A, q)

        w = v/alpha
        for c, qc in zip(cs, q):
            w = axpy(c, w, w.size, -qc)

        return w

    def matvec(self, v):
        """Evaluate w = M v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v):
        """Evaluate w = M^H v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)

    def solve(self, v, tol=0):
        """Evaluate w = M^-1 v"""
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v, tol):
        """Evaluate w = M^-H v"""
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    def append(self, c, d):
        if self.collapsed is not None:
            self.collapsed += c[:,None] * d[None,:].conj()
            return

        self.cs.append(c)
        self.ds.append(d)

        if len(self.cs) > c.size:
            self.collapse()

    def __array__(self):
        if self.collapsed is not None:
            return self.collapsed

        Gm = self.alpha*np.identity(self.n, dtype=self.dtype)
        for c, d in zip(self.cs, self.ds):
            Gm += c[:,None]*d[None,:]
        return Gm

    def collapse(self):
        """Collapse the low-rank matrix to a full-rank one."""
        self.collapsed = np.array(self)
        self.cs = None
        self.ds = None
        self.alpha = None

    def restart_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping all vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    def simple_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]

    def svd_reduce(self, max_rank, to_retain=None):
        """
        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the \"Broyden Rank Reduction Inverse\"
        algorithm described in [vR]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [vR] B.A. van der Rotten, PhD thesis,
           \"A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations\". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).
           
           http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        """
        if self.collapsed is not None:
            return

        p = max_rank
        if to_retain is not None:
            q = to_retain
        else:
            q = p - 2

        if self.cs:
            p = min(p, len(self.cs[0]))
        q = max(0, min(q, p-1))

        m = len(self.cs)
        if m < p:
            # nothing to do
            return

        C = np.array(self.cs).T
        D = np.array(self.ds).T

        D, R = qr(D, mode='qr', econ=True)
        C = dot(C, R.T.conj())

        U, S, WH = svd(C, full_matrices=False, compute_uv=True)

        C = dot(C, inv(WH))
        D = dot(D, WH.T.conj())

        for k in xrange(q):
            self.cs[k] = C[:,k].copy()
            self.ds[k] = D[:,k].copy()

        del self.cs[q:]
        del self.ds[q:]

_doc_parts['broyden_params'] = """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    reduction_method : str or tuple, optional
        Method used in ensuring that the rank of the Broyden matrix
        stays low. Can either be a string giving the name of the method,
        or a tuple of the form ``(method, param1, param2, ...)``
        that gives the name of the method and values for additional parameters.

        Methods available:
            - ``none``: no reduction, allow infinite rank.
            - ``restart``: drop all matrix columns. Has no extra parameters.
            - ``simple``: drop oldest matrix column. Has no extra parameters.
            - ``svd``: keep only the most significant SVD components.
              Extra parameters:
                  - ``to_retain`: number of SVD components to retain when
                    rank reduction is done. Default is ``max_rank - 2``.
    max_rank : int or tuple, optional
        Maximum rank for the Broyden matrix.
    """.strip()

class BroydenFirst(GenericBroyden):
    """
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \"Broyden's good method\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    Notes
    -----
    This implementation of the Broyden method stores the inverse Jacobian.

    """

    def __init__(self, alpha=0.1, reduction_method='none', max_rank=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.Gm = None
        
        if max_rank is None:
            max_rank = np.inf
        self.max_rank = max_rank

        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        reduce_params = (max_rank - 1,) + reduce_params[1:]

        if reduction_method == 'svd':
            self._reduce = lambda: self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda: self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda: self.Gm.restart_reduce(*reduce_params)
        elif reduction_method in ('none', None):
            self._reduce = lambda: None
        else:
            raise ValueError("Unknown rank reduction method '%s'" %
                             reduction_method)

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        return inv(self.Gm)

    def solve(self, f, tol=0):
        return self.Gm.matvec(f)

    def matvec(self, f):
        return self.Gm.solve(f)

    def rsolve(self, f):
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce() # reduce first to preserve secant condition

        s = self.Gm.rmatvec(dx)
        y = self.Gm.matvec(df)
        c = (dx - y) / vdot(dx, y)
        d = s
        self.Gm.append(c, d)
        #self.Gm += (dx - y)[:,None] * s[None,:] / dot(dx, y)

class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    Notes
    -----
    This implementation of the Broyden method stores the inverse Jacobian.

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce() # reduce first to preserve secant condition

        c = (dx - self.Gm.matvec(df)) / df_norm
        d = df / df_norm
        self.Gm.append(c, d)
        #self.Gm += (dx - dot(self.Gm, df))[:,None] * df[None,:]/df_norm**2

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

    # Note:
    #
    # Anderson method maintains a rank M approximation of the inverse Jacobian,
    #
    #     J^-1 v ~ -v*alpha + (dX + alpha dF) A^-1 dF^H v
    #     A      = W + dF^H dF
    #     W      = w0^2 diag(dF^H dF)
    #
    # so that for w0 = 0 the secant condition applies for last M iterates, ie.,
    #
    #     J^-1 df_j = dx_j
    #
    # for all j = 0 ... M-1.
    #
    # Moreover, (from Sherman-Morrison-Woodbury formula)
    #
    #    J v ~ [ b I - b^2 C (I + b dF^H A^-1 C)^-1 dF^H ] v
    #    C   = (dX + alpha dF) A^-1
    #    b   = -1/alpha
    #
    # and after simplification
    #
    #    J v ~ -v/alpha + (dX/alpha + dF) (dF^H dX - alpha W)^-1 dF^H v
    #

    def __init__(self, alpha=0.1, w0=0.01, M=5):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f, tol=0):
        dx = -self.alpha*f

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in xrange(n):
            df_f[k] = vdot(self.df[k], f)
        gamma = solve(self.a, df_f)

        for m in xrange(n):
            dx += gamma[m]*(self.dx[m] + self.alpha*self.df[m])
        return dx

    def matvec(self, f):
        dx = -f/self.alpha

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in xrange(n):
            df_f[k] = vdot(self.df[k], f)

        b = np.empty((n, n), dtype=f.dtype)
        for i in xrange(n):
            for j in xrange(n):
                b[i,j] = vdot(self.df[i], self.dx[j])
                if i == j and self.w0 != 0:
                    b[i,j] -= vdot(self.df[i], self.df[i])*self.w0**2*self.alpha
        gamma = solve(b, df_f)

        for m in xrange(n):
            dx += gamma[m]*(self.df[m] + self.dx[m]/self.alpha)
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
        a = np.zeros((n, n), dtype=f.dtype)

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

    def __init__(self, alpha=0.1):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.d = np.ones((self.shape[0],), dtype=self.dtype) / self.alpha

    def solve(self, f, tol=0):
        return -f / self.d

    def matvec(self, f):
        return -f * self.d

    def rsolve(self, f):
        return -f / self.d.conj()

    def rmatvec(self, f):
        return -f * self.d.conj()

    def todense(self):
        return np.diag(-self.d)

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

    def __init__(self, alpha=0.1):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def solve(self, f, tol=0):
        return -f*self.alpha

    def matvec(self, f):
        return -f/self.alpha

    def rsolve(self, f):
        return -f*np.conj(self.alpha)

    def rmatvec(self, f):
        return -f/np.conj(self.alpha)

    def todense(self):
        return np.diag(-np.ones(self.shape[0])/self.alpha)

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

    def __init__(self, alpha=0.1, alphamax=1.0):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.beta = self.alpha * np.ones((self.shape[0],), dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f*self.beta

    def matvec(self, f):
        return -f/self.beta

    def rsolve(self, f, tol):
        return -f*self.beta.conj()

    def rmatvec(self, f):
        return -f/self.beta.conj()

    def todense(self):
        return np.diag(-1/self.beta)

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
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> jac = Broyden1()
        >>> kjac = KrylovJacobian(inner_M=jac.inverse).
    inner_tol, inner_maxiter, ...
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

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=6, **kw):
        self.preconditioner = inner_M
        self.rdiff = rdiff
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

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

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        sol, info = self.method(self.op, rhs, tol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()

        # Update also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)

        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1./2)

        self._update_diff_step()


        # Setup also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)


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
    kwargs = zip(args[-len(defaults):], defaults)
    kw_str = ", ".join(["%s=%r" % (k, v) for k, v in kwargs])
    if kw_str:
        kw_str = ", " + kw_str
    kwkw_str = ", ".join(["%s=%s" % (k, k) for k, v in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ", "

    # Construct the wrapper function so that it's keyword arguments
    # are visible in pydoc.help etc.
    wrapper = """
def %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None, 
             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, 
             tol_norm=None, line_search='armijo', callback=None, **kw):
    jac = %(jac)s(%(kwkw)s **kw)
    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,
                        callback)
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
