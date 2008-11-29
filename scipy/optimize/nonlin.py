"""
Nonlinear solvers
=================

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

A collection of general-purpose nonlinear multidimensional solvers.

   broyden1            --  Broyden's first method - is a quasi-Newton-Raphson
                           method for updating an approximate Jacobian and then
                           inverting it
   broyden2            --  Broyden's second method - the same as broyden1, but
                           updates the inverse Jacobian directly
   broyden3            --  Broyden's second method - the same as broyden2, but
                           instead of directly computing the inverse Jacobian,
                           it remembers how to construct it using vectors, and
                           when computing inv(J)*F, it uses those vectors to
                           compute this product, thus avoding the expensive NxN
                           matrix multiplication.
   broyden_generalized --  Generalized Broyden's method, the same as broyden2,
                           but instead of approximating the full NxN Jacobian,
                           it construct it at every iteration in a way that
                           avoids the NxN matrix multiplication.  This is not
                           as precise as broyden3.

   anderson            --  extended Anderson method, the same as the
                           broyden_generalized, but added w_0^2*I to before
                           taking inversion to improve the stability
   anderson2           --  the Anderson method, the same as anderson, but
                           formulated differently

"""

import math
import numpy as np
from numpy.linalg import norm, solve
from numpy import asarray, dot

__all__ = ['broyden1', 'broyden2', 'broyden3',
           'broyden1_modified', 'broyden_modified', 'linearmixing',
           'vackar', 'excitingmixing', 'broyden_generalized',
           'anderson', 'anderson2']

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

class GoodBroyden(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using 'Good' Broyden Jacobian approximation.
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

class BadBroyden(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using 'Bad' Broyden Jacobian approximation.
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

class BadBroyden2(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using 'Bad' Broyden Jacobian approximation.
    
    The inverse Jacobian matrix is stored implicitly in this method.
    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).""")
    
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.zy = []
        self.alpha = alpha

    def solve(self, f):
        s = -self.alpha * f
        for z, y in self.zy:
            s += z * dot(y, f)
        return s

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        z = dx - self.solve(df)
        y = df / norm(df)**2
        self.zy.append((z, y))

class BadBroydenModified(BadBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using 'Bad' Broyden Jacobian approximation.
    
    The inverse Jacobian matrix is updated using non-standard matrix
    identities in this method.
    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).""")
    
    def _inv(self, A, u, v):
        Au = dot(A, u)
        return A - Au[:,None] * dot(v,A)[None,:]/(1. + dot(v, Au))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        dx /= dx_norm
        df /= dx_norm
        self.Gm = self._inv(self.Gm + dx[:,None]*dot(dx,self.Gm)[None,:],df,dx)

#------------------------------------------------------------------------------
# Modified formulas; not necessarily Quasi-Newton any more
#------------------------------------------------------------------------------

class BadBroydenModified2(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using 'Bad' Broyden Jacobian approximation.
    
    In this routine, the inverse Jacobian matrix is stored in in implicit form.
    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    w0 : float, optional
        Tunable parameter in the update formula
    wl : float, optionaö
        Tunable parameter in the update formula""")

    def __init__(self, x0, f0, alpha=0.35, w0=0.01, wl=5):
        GenericBroyden.__init__(self, x0, f0)
        self.w = []
        self.u = []
        self.df = []
        self.beta = None
        self.alpha = alpha
        self.wl = wl
        self.w0 = w0

    def solve(self, f):
        w, u, beta = self.w, self.u, self.beta
        dx = -self.alpha * f
        M = len(self.w)
        for i in xrange(M):
            for j in xrange(M):
                dx += w[i]*w[j]*beta[i,j]*u[j]*dot(self.df[i], f)
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        f_norm = norm(f)

        w, u = self.w, self.u

        w.append(self.wl / f_norm)
        u.append((self.alpha*df + dx) / df_norm)
        self.df.append(df / df_norm)

        M = len(self.w)
        a = np.empty((M, M))
        for i in xrange(M):
            for j in xrange(M):
                a[i,j] =w[i]*w[j]*dot(self.df[j], self.df[i])
        self.beta = np.linalg.inv(self.w0**2*np.identity(M) + a)

#------------------------------------------------------------------------------
# Broyden-like (restricted memory)
#------------------------------------------------------------------------------

class BroydenGeneralized(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using a variant of Anderson mixing.
    
    The jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversion and MxN multiplication is required.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.""")

    def __init__(self, x0, f0, alpha=0.1, M=5):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None

    def solve(self, f):
        dx = -self.alpha*f
        for m in xrange(len(self.dx)):
            dx += self.gamma[m]*self.dx[m] + self.alpha*self.df[m]
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
        a = self._form_a()
        
        dFF = np.empty(n)
        for k in xrange(n):
            dFF[k] = dot(self.df[k], f)

        self.gamma = solve(a, dFF)

    def _form_a(self):
        n = len(self.dx)
        a = np.empty((n, n))
        for i in xrange(n):
            for j in xrange(n):
                a[i,j] = dot(self.df[i], self.df[j])
        return a

class Anderson(BroydenGeneralized):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using extended Anderson mixing.
    
    It is formed by for a 'best' solution in the space spanned by last `M`
    vectors. As a result, only a MxM matrix inversion and MxN multiplication
    is required.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter.""")
    
    def __init__(self, x0, f0, alpha=0.1, M=5, w0=0.01):
        BroydenGeneralized.__init__(self, x0, f0, alpha, M)
        self.w0 = w0

    def _form_a(self):
        n = len(self.dx)
        a = np.empty((n, n))
        for i in xrange(n):
            for j in xrange(n):
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1+wd)*dot(self.df[i], self.df[j])
        return a

class Anderson2(GenericBroyden):
    __doc__ = nonlin_solve.__doc__ % ("""
    Find a root of a function, using extended Anderson mixing (another form).
    
    It is formed by for a 'best' solution in the space spanned by last `M`
    vectors. As a result, only a MxM matrix inversion and MxN multiplication
    is required.

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """, """
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter.""")

    def __init__(self, x0, f0, alpha=0.1, M=5, w0=0.01):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.M = M
        self.w0 = w0
        self.df = []

    def solve(self, f):
        dx = f.copy()
        for m in xrange(len(self.df)):
            dx += self.theta[m] * (self.df[m] - f)
        dx *= -self.alpha
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.df.append(f - df)

        while len(self.df) > self.M:
            self.df.pop(0)

        n = len(self.df)
        a = np.empty((n, n))
        for i in xrange(n):
            for j in xrange(n):
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1 + wd)*dot(f - self.df[i], f - self.df[j])

        dFF = np.empty(n)
        for k in xrange(n):
            dFF[k] = dot(f - self.df[k], f)

        self.theta = solve(a, dFF)

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

broyden1 = _broyden_wrapper('broyden1', GoodBroyden)
broyden2 = _broyden_wrapper('broyden2', BadBroyden)
broyden3 = _broyden_wrapper('broyden3', BadBroyden2)
broyden1_modified = _broyden_wrapper('broyden1_modified', BadBroydenModified)
broyden_modified = _broyden_wrapper('broyden_modified', BadBroydenModified2)
linearmixing = _broyden_wrapper('linearmixing', LinearMixing)
vackar = _broyden_wrapper('vackar', Vackar)
excitingmixing = _broyden_wrapper('excitingmixing', ExcitingMixing)
broyden_generalized = _broyden_wrapper('broyden_generalized',
                                       BroydenGeneralized)
anderson = _broyden_wrapper('anderson', Anderson)
anderson2 = _broyden_wrapper('anderson2', Anderson2)
