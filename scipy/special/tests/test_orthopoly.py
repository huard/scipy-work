"""
Orthogonal polynomials
"""

from test_basic import assert_tol_equal
from scipy.special import *

class TestCheby(TestCase):
    def test_chebyc(self):
        C0 = chebyc(0)
        C1 = chebyc(1)
        C2 = chebyc(2)
        C3 = chebyc(3)
        C4 = chebyc(4)
        C5 = chebyc(5)

        assert_array_almost_equal(C0.c,[2],13)
        assert_array_almost_equal(C1.c,[1,0],13)
        assert_array_almost_equal(C2.c,[1,0,-2],13)
        assert_array_almost_equal(C3.c,[1,0,-3,0],13)
        assert_array_almost_equal(C4.c,[1,0,-4,0,2],13)
        assert_array_almost_equal(C5.c,[1,0,-5,0,5,0],13)

    def test_chebys(self):
        S0 = chebys(0)
        S1 = chebys(1)
        S2 = chebys(2)
        S3 = chebys(3)
        S4 = chebys(4)
        S5 = chebys(5)
        assert_array_almost_equal(S0.c,[1],13)
        assert_array_almost_equal(S1.c,[1,0],13)
        assert_array_almost_equal(S2.c,[1,0,-1],13)
        assert_array_almost_equal(S3.c,[1,0,-2,0],13)
        assert_array_almost_equal(S4.c,[1,0,-3,0,1],13)
        assert_array_almost_equal(S5.c,[1,0,-4,0,3,0],13)

    def test_chebyt(self):
        T0 = chebyt(0)
        T1 = chebyt(1)
        T2 = chebyt(2)
        T3 = chebyt(3)
        T4 = chebyt(4)
        T5 = chebyt(5)
        assert_array_almost_equal(T0.c,[1],13)
        assert_array_almost_equal(T1.c,[1,0],13)
        assert_array_almost_equal(T2.c,[2,0,-1],13)
        assert_array_almost_equal(T3.c,[4,0,-3,0],13)
        assert_array_almost_equal(T4.c,[8,0,-8,0,1],13)
        assert_array_almost_equal(T5.c,[16,0,-20,0,5,0],13)

    def test_chebyu(self):
        U0 = chebyu(0)
        U1 = chebyu(1)
        U2 = chebyu(2)
        U3 = chebyu(3)
        U4 = chebyu(4)
        U5 = chebyu(5)
        assert_array_almost_equal(U0.c,[1],13)
        assert_array_almost_equal(U1.c,[2,0],13)
        assert_array_almost_equal(U2.c,[4,0,-1],13)
        assert_array_almost_equal(U3.c,[8,0,-4,0],13)
        assert_array_almost_equal(U4.c,[16,0,-12,0,1],13)
        assert_array_almost_equal(U5.c,[32,0,-32,0,6,0],13)

class TestGegenbauer(TestCase):

    def test_gegenbauer(self):
        a = 5*rand()-0.5
        if any(a==0): a = -0.2
        Ca0 = gegenbauer(0,a)
        Ca1 = gegenbauer(1,a)
        Ca2 = gegenbauer(2,a)
        Ca3 = gegenbauer(3,a)
        Ca4 = gegenbauer(4,a)
        Ca5 = gegenbauer(5,a)

        assert_array_almost_equal(Ca0.c,array([1]),13)
        assert_array_almost_equal(Ca1.c,array([2*a,0]),13)
        assert_array_almost_equal(Ca2.c,array([2*a*(a+1),0,-a]),13)
        assert_array_almost_equal(Ca3.c,array([4*poch(a,3),0,-6*a*(a+1),
                                               0])/3.0,11)
        assert_array_almost_equal(Ca4.c,array([4*poch(a,4),0,-12*poch(a,3),
                                               0,3*a*(a+1)])/6.0,11)
        assert_array_almost_equal(Ca5.c,array([4*poch(a,5),0,-20*poch(a,4),
                                               0,15*poch(a,3),0])/15.0,11)


class TestHermite(TestCase):
    def test_hermite(self):
        H0 = hermite(0)
        H1 = hermite(1)
        H2 = hermite(2)
        H3 = hermite(3)
        H4 = hermite(4)
        H5 = hermite(5)
        assert_array_almost_equal(H0.c,[1],13)
        assert_array_almost_equal(H1.c,[2,0],13)
        assert_array_almost_equal(H2.c,[4,0,-2],13)
        assert_array_almost_equal(H3.c,[8,0,-12,0],13)
        assert_array_almost_equal(H4.c,[16,0,-48,0,12],12)
        assert_array_almost_equal(H5.c,[32,0,-160,0,120,0],12)

    def test_hermitenorm(self):
        # He_n(x) = 2**(-n/2) H_n(x/sqrt(2))
        psub = poly1d([1.0/sqrt(2),0])
        H0 = hermitenorm(0)
        H1 = hermitenorm(1)
        H2 = hermitenorm(2)
        H3 = hermitenorm(3)
        H4 = hermitenorm(4)
        H5 = hermitenorm(5)
        he0 = hermite(0)(psub)
        he1 = hermite(1)(psub) / sqrt(2)
        he2 = hermite(2)(psub) / 2.0
        he3 = hermite(3)(psub) / (2*sqrt(2))
        he4 = hermite(4)(psub) / 4.0
        he5 = hermite(5)(psub) / (4.0*sqrt(2))

        assert_array_almost_equal(H0.c,he0.c,13)
        assert_array_almost_equal(H1.c,he1.c,13)
        assert_array_almost_equal(H2.c,he2.c,13)
        assert_array_almost_equal(H3.c,he3.c,13)
        assert_array_almost_equal(H4.c,he4.c,13)
        assert_array_almost_equal(H5.c,he5.c,13)


class TestLaguerre(TestCase):
    def test_laguerre(self):
        lag0 = laguerre(0)
        lag1 = laguerre(1)
        lag2 = laguerre(2)
        lag3 = laguerre(3)
        lag4 = laguerre(4)
        lag5 = laguerre(5)
        assert_array_almost_equal(lag0.c,[1],13)
        assert_array_almost_equal(lag1.c,[-1,1],13)
        assert_array_almost_equal(lag2.c,array([1,-4,2])/2.0,13)
        assert_array_almost_equal(lag3.c,array([-1,9,-18,6])/6.0,13)
        assert_array_almost_equal(lag4.c,array([1,-16,72,-96,24])/24.0,13)
        assert_array_almost_equal(lag5.c,array([-1,25,-200,600,-600,120])/120.0,13)

    def test_genlaguerre(self):
        k = 5*rand()-0.9
        lag0 = genlaguerre(0,k)
        lag1 = genlaguerre(1,k)
        lag2 = genlaguerre(2,k)
        lag3 = genlaguerre(3,k)
        assert_equal(lag0.c,[1])
        assert_equal(lag1.c,[-1,k+1])
        assert_almost_equal(lag2.c,array([1,-2*(k+2),(k+1.)*(k+2.)])/2.0)
        assert_almost_equal(lag3.c,array([-1,3*(k+3),-3*(k+2)*(k+3),(k+1)*(k+2)*(k+3)])/6.0)


class TestLegendre(TestCase):
    def test_legendre(self):
        leg0 = legendre(0)
        leg1 = legendre(1)
        leg2 = legendre(2)
        leg3 = legendre(3)
        leg4 = legendre(4)
        leg5 = legendre(5)
        assert_equal(leg0.c,[1])
        assert_equal(leg1.c,[1,0])
        assert_equal(leg2.c,array([3,0,-1])/2.0)
        assert_almost_equal(leg3.c,array([5,0,-3,0])/2.0)
        assert_almost_equal(leg4.c,array([35,0,-30,0,3])/8.0)
        assert_almost_equal(leg5.c,array([63,0,-70,0,15,0])/8.0)

class TestJacobi(TestCase):
    def test_jacobi(self):
        a = 5*rand() - 1
        b = 5*rand() - 1
        P0 = jacobi(0,a,b)
        P1 = jacobi(1,a,b)
        P2 = jacobi(2,a,b)
        P3 = jacobi(3,a,b)

        assert_array_almost_equal(P0.c,[1],13)
        assert_array_almost_equal(P1.c,array([a+b+2,a-b])/2.0,13)
        cp = [(a+b+3)*(a+b+4), 4*(a+b+3)*(a+2), 4*(a+1)*(a+2)]
        p2c = [cp[0],cp[1]-2*cp[0],cp[2]-cp[1]+cp[0]]
        assert_array_almost_equal(P2.c,array(p2c)/8.0,13)
        cp = [(a+b+4)*(a+b+5)*(a+b+6),6*(a+b+4)*(a+b+5)*(a+3),
              12*(a+b+4)*(a+2)*(a+3),8*(a+1)*(a+2)*(a+3)]
        p3c = [cp[0],cp[1]-3*cp[0],cp[2]-2*cp[1]+3*cp[0],cp[3]-cp[2]+cp[1]-cp[0]]
        assert_array_almost_equal(P3.c,array(p3c)/48.0,13)


class TestShLegendre(TestCase):
    def test_sh_legendre(self):
        # P*_n(x) = P_n(2x-1)
        psub = poly1d([2,-1])
        Ps0 = sh_legendre(0)
        Ps1 = sh_legendre(1)
        Ps2 = sh_legendre(2)
        Ps3 = sh_legendre(3)
        Ps4 = sh_legendre(4)
        Ps5 = sh_legendre(5)
        pse0 = legendre(0)(psub)
        pse1 = legendre(1)(psub)
        pse2 = legendre(2)(psub)
        pse3 = legendre(3)(psub)
        pse4 = legendre(4)(psub)
        pse5 = legendre(5)(psub)
        assert_array_almost_equal(Ps0.c,pse0.c,13)
        assert_array_almost_equal(Ps1.c,pse1.c,13)
        assert_array_almost_equal(Ps2.c,pse2.c,13)
        assert_array_almost_equal(Ps3.c,pse3.c,13)
        assert_array_almost_equal(Ps4.c,pse4.c,12)
        assert_array_almost_equal(Ps5.c,pse5.c,12)


class TestShChebyt(TestCase):
    def test_sh_chebyt(self):
        # T*_n(x) = T_n(2x-1)
        psub = poly1d([2,-1])
        Ts0 = sh_chebyt(0)
        Ts1 = sh_chebyt(1)
        Ts2 = sh_chebyt(2)
        Ts3 = sh_chebyt(3)
        Ts4 = sh_chebyt(4)
        Ts5 = sh_chebyt(5)
        tse0 = chebyt(0)(psub)
        tse1 = chebyt(1)(psub)
        tse2 = chebyt(2)(psub)
        tse3 = chebyt(3)(psub)
        tse4 = chebyt(4)(psub)
        tse5 = chebyt(5)(psub)
        assert_array_almost_equal(Ts0.c,tse0.c,13)
        assert_array_almost_equal(Ts1.c,tse1.c,13)
        assert_array_almost_equal(Ts2.c,tse2.c,13)
        assert_array_almost_equal(Ts3.c,tse3.c,13)
        assert_array_almost_equal(Ts4.c,tse4.c,12)
        assert_array_almost_equal(Ts5.c,tse5.c,12)


class TestShChebyu(TestCase):
    def test_sh_chebyu(self):
        # U*_n(x) = U_n(2x-1)
        psub = poly1d([2,-1])
        Us0 = sh_chebyu(0)
        Us1 = sh_chebyu(1)
        Us2 = sh_chebyu(2)
        Us3 = sh_chebyu(3)
        Us4 = sh_chebyu(4)
        Us5 = sh_chebyu(5)
        use0 = chebyu(0)(psub)
        use1 = chebyu(1)(psub)
        use2 = chebyu(2)(psub)
        use3 = chebyu(3)(psub)
        use4 = chebyu(4)(psub)
        use5 = chebyu(5)(psub)
        assert_array_almost_equal(Us0.c,use0.c,13)
        assert_array_almost_equal(Us1.c,use1.c,13)
        assert_array_almost_equal(Us2.c,use2.c,13)
        assert_array_almost_equal(Us3.c,use3.c,13)
        assert_array_almost_equal(Us4.c,use4.c,12)
        assert_array_almost_equal(Us5.c,use5.c,11)

class TestShJacobi(TestCase):

    def test_sh_jacobi(self):
        # G^(p,q)_n(x) = n! gamma(n+p)/gamma(2*n+p) * P^(p-q,q-1)_n(2*x-1)
        conv = lambda n,p: _gam(n+1)*_gam(n+p)/_gam(2*n+p)
        psub = poly1d([2,-1])
        q = 4*rand()
        p = q-1 + 2*rand()
        #print "shifted jacobi p,q = ", p, q
        G0 = sh_jacobi(0,p,q)
        G1 = sh_jacobi(1,p,q)
        G2 = sh_jacobi(2,p,q)
        G3 = sh_jacobi(3,p,q)
        G4 = sh_jacobi(4,p,q)
        G5 = sh_jacobi(5,p,q)
        ge0 = jacobi(0,p-q,q-1)(psub) * conv(0,p)
        ge1 = jacobi(1,p-q,q-1)(psub) * conv(1,p)
        ge2 = jacobi(2,p-q,q-1)(psub) * conv(2,p)
        ge3 = jacobi(3,p-q,q-1)(psub) * conv(3,p)
        ge4 = jacobi(4,p-q,q-1)(psub) * conv(4,p)
        ge5 = jacobi(5,p-q,q-1)(psub) * conv(5,p)

        assert_array_almost_equal(G0.c,ge0.c,13)
        assert_array_almost_equal(G1.c,ge1.c,13)
        assert_array_almost_equal(G2.c,ge2.c,13)
        assert_array_almost_equal(G3.c,ge3.c,13)
        assert_array_almost_equal(G4.c,ge4.c,13)
        assert_array_almost_equal(G5.c,ge5.c,13)
