#!/usr/bin/env python
#
# Created by: Pearu Peterson, September 2002
#

__usage__ = """
Build lapack:
  python setup_lapack.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.lib.lapack.test(<level>)'
Run tests if lapack is not installed:
  python tests/test_lapack.py [<level>]
"""

import sys
from scipy_test.testing import *
from scipy_base import ones
set_package_path()
from lapack import flapack
clapack = None
#from lapack import clapack
restore_path()

class test_flapack_simple(ScipyTestCase):

    def check_gebal(self):
        a = [[1,2,3],[4,5,6],[7,8,9]]
        a1 = [[1,0,0,3e-4],
              [4,0,0,2e-3],
              [7,1,0,0],
              [0,1,0,0]]
        for p in 'sdzc':
            f = getattr(flapack,p+'gebal',None)
            if f is None: continue
            ba,lo,hi,pivscale,info = f(a)
            assert not info,`info`
            assert_array_almost_equal(ba,a)
            assert_equal((lo,hi),(0,len(a[0])-1))
            assert_array_almost_equal(pivscale,ones(len(a)))

            ba,lo,hi,pivscale,info = f(a1,permute=1,scale=1)
            assert not info,`info`

    def check_gehrd(self):
        a = [[-149, -50,-154],
             [ 537, 180, 546],
             [ -27,  -9, -25]]
        for p in 'sdzc':
            f = getattr(flapack,p+'gehrd',None)
            if f is None: continue
            ht,tau,info = f(a)
            assert not info,`info`

class test_lapack(ScipyTestCase):

    def check_flapack(self):
        if hasattr(flapack,'empty_module'):
            print """
****************************************************************
WARNING: flapack module is empty
-----------
See scipy/INSTALL.txt for troubleshooting.
****************************************************************
"""
    def check_clapack(self):
        if hasattr(clapack,'empty_module'):
            print """
****************************************************************
WARNING: clapack module is empty
-----------
See scipy/INSTALL.txt for troubleshooting.
Notes:
* If atlas library is not found by scipy/system_info.py,
  then scipy uses flapack instead of clapack.
****************************************************************
"""

if __name__ == "__main__":
    ScipyTest().run()
