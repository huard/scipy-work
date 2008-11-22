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

The broyden2 is the best. For large systems, use broyden3. excitingmixing is
also very effective. There are some more solvers implemented (see their
docstrings), however, those are of mediocre quality.

"""

import math
import numpy as np
from numpy.linalg import norm, solve
from numpy import asarray

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def a_F(F,xm):
    return asarray(F(asarray(xm).ravel())).ravel()

def m_F(F,xm):
    return np.asmatrix(a_F(F, xm)).T

def maxnorm(x):
    return np.absolute(x).max()

class NoConvergence(Exception):
    pass

class TerminationCondition(object):
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """
    def __init__(self, f_tol=None, f_rtol=np.inf, x_tol=np.inf, x_rtol=np.inf,
                 iter=None, norm=maxnorm):
        if f_tol is None and iter is None:
            f_tol = 1e-6
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

        if self.iter is not None and self.iteration > self.iter:
            # backwards compatibility with Scipy 0.6.0
            return True

        # NB: condition must succeed for x_rtol=infty even if x_norm == 0
        return ((f_norm <= self.f_tol and f_norm/self.f_rtol <= self.f0_norm)
                and (dx_norm <= self.x_tol and dx_norm/self.x_rtol <= x_norm))

def _default_maxiter(maxiter, iter, xm):
    """Sensible default number of maximum iterations, depending on input"""
    if maxiter is not None:
        return maxiter
    elif iter is not None:
        return iter + 1
    else:
        return 100*(xm.size + 1)

def _as_inexact(x):
    """Return `x` as an array, of either floats or complex floats"""
    x = asarray(x)
    if not isinstance(x.dtype.type, np.inexact):
        return asarray(x, dtype=np.float_)
    return x

#------------------------------------------------------------------------------
# (Full) Broyden variants
#------------------------------------------------------------------------------

def broyden1(F, xin, iter=None, alpha=0.1, verbose = False, maxiter=None,
             **kw):
    """Broyden's first method.

    Updates Jacobian and computes inv(J) by a matrix inversion at every
    iteration. It's very slow.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    Jm=-1/alpha*np.asmatrix(np.identity(len(xm)))
    deltaxm = np.inf

    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=solve(-Jm,Fxm)
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        Jm=Jm+(deltaFxm-Jm*deltaxm)*deltaxm.T/norm(deltaxm)**2
        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def broyden2(F, xin, iter=None, alpha=0.4, verbose=False,
             maxiter=None, **kw):
    """Broyden's second method.

    Updates inverse Jacobian by an optimal formula.
    There is NxN matrix multiplication in every iteration.

    Recommended.
    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    Gm=-alpha*np.asmatrix(np.identity(len(xm)))
    deltaxm = np.inf

    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=-Gm*Fxm
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        Gm=Gm+(deltaxm-Gm*deltaFxm)*deltaFxm.T/norm(deltaFxm)**2
        if verbose:
            print "%d:  |F(x)|=%g"%(condition.iteration, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def broyden3(F, xin, iter=None, alpha=0.4, verbose=False,
             maxiter=None, **kw):
    """Broyden's second method.

    Updates inverse Jacobian by an optimal formula.
    The NxN matrix multiplication is avoided.

    Recommended.
    """
    condition = TerminationCondition(iter=iter, **kw)
    
    zy=[]
    def updateG(z,y):
        "G:=G+z*y.T"
        zy.append((z,y))
    def Gmul(f):
        "G=-alpha*1+z*y.T+z*y.T ..."
        s=-alpha*f
        for z,y in zy:
            s=s+z*(y.T*f)
        return s
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
#    Gm=-alpha*np.asmatrix(np.identity(len(xm)))
    deltaxm = np.inf

    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        #deltaxm=-Gm*Fxm
        deltaxm=Gmul(-Fxm)
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        #Gm=Gm+(deltaxm-Gm*deltaFxm)*deltaFxm.T/norm(deltaFxm)**2
        updateG(deltaxm-Gmul(deltaFxm),deltaFxm/norm(deltaFxm)**2)
        if verbose:
            print "%d:  |F(x)|=%g"%(condition.iteration, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def broyden1_modified(F, xin, iter=None, alpha=0.1, verbose = False,
                      maxiter=None, **kw):
    """Broyden's first method, modified by O. Certik.

    Updates inverse Jacobian using some matrix identities at every iteration,
    its faster then newton_slow, but still not optimal.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    def inv(A,u,v):
        #interesting is that this
        #return (A.I+u*v.T).I
        #is more stable than
        #return A-A*u*v.T*A/float(1+v.T*A*u)
        Au=A*u
        return A-Au*(v.T*A)/float(1+v.T*Au)
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    deltaxm = np.inf
    Jm=alpha*np.asmatrix(np.identity(len(xm)))
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=Jm*Fxm
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
#        print "-------------",norm(deltaFxm),norm(deltaxm)
        deltaFxm/=norm(deltaxm)
        deltaxm/=norm(deltaxm)
        Jm=inv(Jm+deltaxm*deltaxm.T*Jm,-deltaFxm,deltaxm)

        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def broyden_modified(F, xin, iter=None, alpha=0.35, w0=0.01, wl=5,
                     verbose=False, maxiter=None, **kw):
    """Modified Broyden's method.

    Updates inverse Jacobian using information from all the iterations and
    avoiding the NxN matrix multiplication. The problem is with the weights,
    it converges the same or worse than broyden2 or broyden_generalized

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.matrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    G0=alpha
    w=[]
    u=[]
    dFxm=[]
    deltaxm = np.inf
    
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=G0*Fxm
        M = len(w)
        for i in range(M):
            for j in range(M):
                deltaxm-=w[i]*w[j]*betta[i,j]*u[j]*(dFxm[i].T*Fxm)
        xm+=deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1

        Fxm_norm = norm(Fxm)
        deltaFxm_norm = norm(deltaFxm)

        if Fxm_norm == 0:
            # converged
            break

        if deltaFxm_norm == 0:
            # XXX: what to do now?
            pass

        w.append(wl/Fxm_norm)
        u.append((G0*deltaFxm+deltaxm)/deltaFxm_norm)
        dFxm.append(deltaFxm/deltaFxm_norm)
        M = len(w)
        a=np.asmatrix(np.empty((M,M)))
        for i in range(M):
            for j in range(M):
                a[i,j]=w[i]*w[j]*dFxm[j].T*dFxm[i]
        betta=(w0**2*np.asmatrix(np.identity(M))+a).I

        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

#------------------------------------------------------------------------------
# Other algorithms
#------------------------------------------------------------------------------

def broyden_generalized(F, xin, iter=None, alpha=0.1, M=5, verbose=False,
                        maxiter=None, **kw):
    """Generalized Broyden's method.

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
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    G0=-alpha
    dxm=[]
    dFxm=[]
    deltaxm=np.inf
    
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=-G0*Fxm
        if M>0:
            MM=min(M,n)
            for m in range(n-MM,n):
                deltaxm=deltaxm-(float(gamma[m-(n-MM)])*dxm[m]-G0*dFxm[m])
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1

        if M>0:
            dxm.append(deltaxm)
            dFxm.append(deltaFxm)
            MM=min(M,n+1)
            a=np.asmatrix(np.empty((MM,MM)))
            for i in range(n+1-MM,n+1):
                for j in range(n+1-MM,n+1):
                    a[i-(n+1-MM),j-(n+1-MM)]=dFxm[i].T*dFxm[j]

            dFF=np.asmatrix(np.empty(MM)).T
            for k in range(n+1-MM,n+1):
                dFF[k-(n+1-MM)]=dFxm[k].T*Fxm
            gamma=a.I*dFF

        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def anderson(F, xin, iter=None, alpha=0.1, M=5, w0=0.01, verbose=False,
             maxiter=None, **kw):
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
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    dxm=[]
    dFxm=[]
    deltaxm = np.inf

    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break

        deltaxm=alpha*Fxm
        if M>0:
            MM=min(M,n)
            for m in range(n-MM,n):
                deltaxm=deltaxm-(float(gamma[m-(n-MM)])*dxm[m]+alpha*dFxm[m])
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        
        if M>0:
            dxm.append(deltaxm)
            dFxm.append(deltaFxm)
            MM=min(M,n+1)
            a=np.asmatrix(np.empty((MM,MM)))
            for i in range(n+1-MM,n+1):
                for j in range(n+1-MM,n+1):
                    if i==j: wd=w0**2
                    else: wd=0
                    a[i-(n+1-MM),j-(n+1-MM)]=(1+wd)*dFxm[i].T*dFxm[j]

            dFF=np.asmatrix(np.empty(MM)).T
            for k in range(n+1-MM,n+1):
                dFF[k-(n+1-MM)]=dFxm[k].T*Fxm
            gamma=solve(a,dFF)
#            print gamma

        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def anderson2(F, xin, iter=None, alpha=0.1, M=5, w0=0.01, verbose = False,
              maxiter=None, **kw):
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
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.asmatrix(_as_inexact(xin)).T
    Fxm=m_F(F,xm)
    dFxm=[]
    deltaxm = np.inf
    
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=Fxm
        if M>0:
            MM=min(M,n)
            for m in range(n-MM,n):
                deltaxm=deltaxm+float(theta[m-(n-MM)])*(dFxm[m]-Fxm)
        deltaxm=deltaxm*alpha
        xm=xm+deltaxm
        Fxm1=m_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1

        if M>0:
            dFxm.append(Fxm-deltaFxm)
            MM=min(M,n+1)
            a=np.asmatrix(np.empty((MM,MM)))
            for i in range(n+1-MM,n+1):
                for j in range(n+1-MM,n+1):
                    if i==j: wd=w0**2
                    else: wd=0
                    a[i-(n+1-MM),j-(n+1-MM)]= \
                        (1+wd)*(Fxm-dFxm[i]).T*(Fxm-dFxm[j])

            dFF=np.asmatrix(np.empty(MM)).T
            for k in range(n+1-MM,n+1):
                dFF[k-(n+1-MM)]=(Fxm-dFxm[k]).T*Fxm
            theta=solve(a,dFF)
#            print gamma

        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def vackar(F, xin, iter=None, alpha=0.1, verbose = False, maxiter=None, **kw):
    """J=diag(d1,d2,...,dN)

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=_as_inexact(xin).ravel()
    Fxm=a_F(F,xm)
    deltaxm = np.inf
    d=1/alpha*np.ones(xm.size)

    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=1/d*Fxm
        xm=xm+deltaxm
        Fxm1=a_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        d=d-(deltaFxm+d*deltaxm)*deltaxm/norm(deltaxm)**2
        if verbose:
            print "%d:  |F(x)|=%g"%(n, norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def linearmixing(F,xin, iter=None, alpha=0.1, verbose=False,
                 maxiter=None, **kw):
    """J=-1/alpha

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=_as_inexact(xin).ravel()
    Fxm=a_F(F,xm)
    deltaxm = np.inf
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=alpha*Fxm
        xm=xm+deltaxm
        Fxm1=a_F(F,xm)
        deltaFxm=Fxm1-Fxm
        Fxm=Fxm1
        if verbose:
            print "%d: |F(x)|=%g" %(n,norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()

def excitingmixing(F, xin, iter=None, alpha=0.1, alphamax=1.0, verbose=False,
                   maxiter=None, **kw):
    """J=-1/alpha

    .. warning::

       The algorithm implemented in this routine is not suitable for
       general root finding. It may be useful for specific problems,
       but whether it will work may depend strongly on the problem.

    """
    condition = TerminationCondition(iter=iter, **kw)
    
    xm=np.array(xin)
    beta=np.array([alpha]*xm.size)
    Fxm=a_F(F,xm)
    deltaxm = np.inf
    maxiter = _default_maxiter(maxiter, iter, xm)
    for n in xrange(maxiter):
        if condition.check(Fxm, xm, deltaxm):
            break
        deltaxm=beta*Fxm
        xm=xm+deltaxm
        Fxm1=a_F(F,xm)
        deltaFxm=Fxm1-Fxm
        for i in range(len(xm)):
            if Fxm1[i]*Fxm[i] > 0:
                beta[i]=beta[i]+alpha
                if beta[i] > alphamax:
                    beta[i] = alphamax
            else:
                beta[i]=alpha
        Fxm=Fxm1
        if verbose:
            print "%d: |F(x)|=%g" %(n,norm(Fxm))
    else:
        raise NoConvergence(asarray(xm).ravel())
    return asarray(xm).ravel()
