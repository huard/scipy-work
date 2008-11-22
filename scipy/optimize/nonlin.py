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
                 tol_norm=None):
    """
    Find a root for a nonlinear function, using a given Jacobian approximation.

    The Jacobian approximation is updated on every iteration, so e.g.
    Quasi-Newton methods are accommodated.

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

        dx = jacobian.get_step()
        x += dx
        Fx = func(x)
        jacobian.update(x.copy(), Fx)

        if verbose or True:
            print "%d:  |F(x)|=%g" % (n, norm(Fx))
    else:
        raise NoConvergence(_array_like(x, x0))

    return _array_like(x, x0)

class Jacobian(object):
    def __init__(self, x0, f0, **kw):
        raise NotImplementedError
    def get_step(self):
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
# Full Broyden variants
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
    """The 'Good' Broyden method; updating the Jacobian"""
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.Jm = (-1./alpha) * np.identity(x0.size)

    def get_step(self):
        return -solve(self.Jm, self.last_f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.Jm += (df - dot(self.Jm, dx))[:,None] * dx[None,:]/dx_norm**2

class BadBroyden(GenericBroyden):
    """The 'Bad' Broyden method; updating the Jacobian inverse"""
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.Gm = -alpha * np.identity(x0.size)

    def get_step(self):
        return -dot(self.Gm, self.last_f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.Gm += (dx - dot(self.Gm, df))[:,None] * df[None,:]/df_norm**2

class BadBroyden2(GenericBroyden):
    """Updating inverse Jacobian, but not storing the explicit matrix"""
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.zy = []
        self.alpha = alpha

    def _G_mul(self, f):
        s = -self.alpha * f
        for z, y in self.zy:
            s += z * dot(y, f)
        return s
    
    def get_step(self):
        return -self._G_mul(self.last_f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        z = dx - self._G_mul(df)
        y = df / norm(df)**2
        self.zy.append((z, y))

class BadBroydenModified(BadBroyden):
    """Updates inverse Jacobian using some matrix identities at every iteration.
    """
    def _inv(self, A, u, v):
        Au = dot(A, u)
        return A - Au[:,None] * dot(v,A)[None,:]/(1. + dot(v, Au))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        dx /= dx_norm
        df /= dx_norm
        self.Gm = self._inv(self.Gm + dx[:,None]*dot(dx,self.Gm)[None,:],df,dx)

class BadBroydenModified2(GenericBroyden):
    """
    Updates inverse Jacobian using information from all the iterations and
    avoiding the NxN matrix multiplication. The problem is with the weights,
    it converges the same or worse than broyden2 or broyden_generalized
    """
    def __init__(self, x0, f0, alpha=0.35, w0=0.01, wl=5):
        GenericBroyden.__init__(self, x0, f0)
        self.w = []
        self.u = []
        self.df = []
        self.beta = None
        self.alpha = alpha
        self.wl = wl
        self.w0 = w0

    def get_step(self):
        w, u, beta, f = self.w, self.u, self.beta, self.last_f
        dx = self.alpha * f
        M = len(self.w)
        for i in xrange(M):
            for j in xrange(M):
                dx -= w[i]*w[j]*beta[i,j]*u[j]*dot(self.df[i], f)
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
    """Generalized Broyden's method (variant of Anderson method).

    Computes an approximation to the inverse Jacobian from the last M
    interations. Avoids NxN matrix multiplication, it only has MxM matrix
    multiplication and inversion.

    M=0 .... linear mixing
    M=1 .... Anderson mixing with 2 iterations
    M=2 .... Anderson mixing with 3 iterations
    etc.
    optimal is M=5

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """

    def __init__(self, x0, f0, alpha=0.1, M=5):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None

    def get_step(self):
        f = self.last_f
        dx = self.alpha*f
        for m in xrange(len(self.dx)):
            dx -= self.gamma[m]*self.dx[m] + self.alpha*self.df[m]
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
    """Extended Anderson method.

    Computes an approximation to the inverse Jacobian from the last M
    interations. Avoids NxN matrix multiplication, it only has MxM matrix
    multiplication and inversion.

    M=0 .... linear mixing
    M=1 .... Anderson mixing with 2 iterations
    M=2 .... Anderson mixing with 3 iterations
    etc.
    optimal is M=5

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    
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
    """Anderson method.

    M=0 .... linear mixing
    M=1 .... Anderson mixing with 2 iterations
    M=2 .... Anderson mixing with 3 iterations
    etc.
    optimal is M=5

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """

    def __init__(self, x0, f0, alpha=0.1, M=5, w0=0.01):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.M = M
        self.w0 = w0
        self.df = []

    def get_step(self):
        dx = self.last_f.copy()
        for m in xrange(len(self.df)):
            dx += self.theta[m] * (self.df[m] - self.last_f)
        dx *= self.alpha
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
    """J=diag(d1,d2,...,dN)

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.d = np.ones_like(x0)/alpha

    def get_step(self):
        return self.last_f / self.d

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d*dx)*dx/dx_norm**2

class LinearMixing(GenericBroyden):
    """J=-1/alpha

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    def __init__(self, x0, f0, alpha=0.1):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha

    def get_step(self):
        return self.last_f*self.alpha

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass

class ExcitingMixing(GenericBroyden):
    """J=-1/alpha

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    def __init__(self, x0, f0, alpha=0.1, alphamax=1.0):
        GenericBroyden.__init__(self, x0, f0)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = alpha*np.ones_like(x0)

    def get_step(self):
        return self.last_f*self.beta

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
               "tol_norm=None):\n"
               "    jac = lambda x, f: %s(x, f %s)\n"
               "    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,\n"
               "        f_tol, f_rtol, x_tol, x_rtol, tol_norm)")
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
