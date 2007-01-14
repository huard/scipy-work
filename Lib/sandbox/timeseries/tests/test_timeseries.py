# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id$
"""
__author__ = "Pierre GF Gerard-Marchant ($Author$)"
__version__ = '1.0'
__revision__ = "$Revision$"
__date__     = '$Date$'

import types

import numpy as N
from numpy import bool_, complex_, float_, int_, object_
import numpy.core.fromnumeric  as fromnumeric
import numpy.core.numeric as numeric
from numpy.testing import NumpyTest, NumpyTestCase
from numpy.testing.utils import build_err_msg

import maskedarray
from maskedarray import masked_array, masked, nomask

import maskedarray.testutils
#reload(maskedarray.testutils)
from maskedarray.testutils import assert_equal, assert_array_equal

#import tdates
##reload(tdates)
#from tdates import date_array_fromlist
import tseries
#reload(tseries)
from tseries import Date, date_array_fromlist
from tseries import time_series, TimeSeries, adjust_endpoints, mask_period

class test_creation(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1,16)]
        dates = date_array_fromlist(dlist)
        data = masked_array(numeric.arange(15), mask=[1,0,0,0,0]*3)
        self.d = (dlist, dates, data)

    def test_fromlist (self):
        "Base data definition."
        (dlist, dates, data) = self.d
        series = time_series(data, dlist)
        assert(isinstance(series, TimeSeries))
        assert_equal(series._mask, [1,0,0,0,0]*3)
        assert_equal(series._series, data)
        assert_equal(series._dates, date_array_fromlist(dlist))
        assert_equal(series.freq, 'D')

    def test_fromrange (self):
        "Base data definition."
        (dlist, dates, data) = self.d
        series = time_series(data, start_date=Date('D',value=dates[0]),
                             length=15)
        assert(isinstance(series, TimeSeries))
        assert_equal(series._mask, [1,0,0,0,0]*3)
        assert_equal(series._series, data)
        assert_equal(series._dates, dates)
        assert_equal(series.freq, 'D')

    def test_fromseries (self):
        "Base data definition."
        (dlist, dates, data) = self.d
        series = time_series(data, dlist)
        dates = dates+15
        series = time_series(series, dates)
        assert(isinstance(series, TimeSeries))
        assert_equal(series._mask, [1,0,0,0,0]*3)
        assert_equal(series._series, data)
        assert_equal(series._dates, dates)
        assert_equal(series.freq, 'D')
#...............................................................................

class test_arithmetics(NumpyTestCase):
    "Some basic arithmetic tests"
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1,16)]
        dates = date_array_fromlist(dlist)
        data = masked_array(numeric.arange(15), mask=[1,0,0,0,0]*3)
        self.d = (time_series(data, dlist), data)
        
    def test_intfloat(self):
        "Test arithmetic timeseries/integers"
        (series, data) =self.d
        #
        nseries = series+1
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data+1)
        assert_equal(nseries._dates, series._dates)
        #        
        nseries = series-1
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data-1)
        assert_equal(nseries._dates, series._dates)
        #
        nseries = series*1
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data*1)
        assert_equal(nseries._dates, series._dates)
        #
        nseries = series/1.
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data/1.)
        assert_equal(nseries._dates, series._dates)
    
    def test_intfloat_inplace(self):
        "Test int/float arithmetics in place."
        (series, data) =self.d
        nseries = series.astype(float_)
        idini = id(nseries)
        data = data.astype(float_)
        #
        nseries += 1.
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data+1.)
        assert_equal(nseries._dates, series._dates)
        assert_equal(id(nseries),idini)
        #
        nseries -= 1.
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data)
        assert_equal(nseries._dates, series._dates)
        assert_equal(id(nseries),idini)
        #
        nseries *= 2.
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data*2.)
        assert_equal(nseries._dates, series._dates)
        assert_equal(id(nseries),idini)
        #
        nseries /= 2.
        assert(isinstance(nseries, TimeSeries))
        assert_equal(nseries._mask, [1,0,0,0,0]*3)
        assert_equal(nseries._series, data)
        assert_equal(nseries._dates, series._dates)
        assert_equal(id(nseries),idini)
    #
    def test_updatemask(self):
        "Checks modification of mask."
        (series, data) =self.d
        assert_equal(series._mask, [1,0,0,0,0]*3)
        series.mask = nomask
        assert(series._mask is nomask)
        assert(series._series._mask is nomask)
        series._series.mask = [1,0,0]*5
        assert_equal(series._mask, [1,0,0]*5)
        assert_equal(series._series._mask, [1,0,0]*5)
        series[2] = masked
        assert_equal(series._mask, [1,0,1]+[1,0,0]*4)
        assert_equal(series._series._mask, [1,0,1]+[1,0,0]*4)
#...............................................................................

class test_getitem(NumpyTestCase):
    "Some getitem tests"
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1,16)]
        dates = date_array_fromlist(dlist)
        data = masked_array(numeric.arange(15), mask=[1,0,0,0,0]*3, dtype=float_)
        self.d = (time_series(data, dlist), data, dates)
    
    def test_wdate(self):
        "Tests  getitem with date as index"
        (series, data, dates) = self.d
        last_date = series[-1]._dates
        assert_equal(series[-1], series[last_date])
        assert_equal(series._dates[-1], dates[-1])
        assert_equal(series[-1]._dates, dates[-1])
        assert_equal(series[last_date]._dates, dates[-1])
        assert_equal(series._series[-1], data._data[-1])
        assert_equal(series[-1]._series, data._data[-1])
        assert_equal(series._mask[-1], data._mask[-1])
        #
        series['2007-01-06'] = 999
        assert_equal(series[5], 999)
        #
    def test_wtimeseries(self):
        "Tests getitem w/ TimeSeries as index"
        (series, data, dates) = self.d
        # Testing a basic condition on data
        cond = (series<8).filled(False)
        dseries = series[cond]
        assert_equal(dseries._data, [1,2,3,4,6,7])
        assert_equal(dseries._dates, series._dates[[1,2,3,4,6,7]])
        assert_equal(dseries._mask, nomask)
        # Testing a basic condition on dates
        series[series._dates < Date('D',string='2007-01-06')] = masked
        assert_equal(series[:5]._series._mask, [1,1,1,1,1])
    
    def test_wslices(self):
        "Test get/set items."
        (series, data, dates) = self.d
        # Basic slices
        assert_equal(series[3:7]._series._data, data[3:7]._data)
        assert_equal(series[3:7]._series._mask, data[3:7]._mask)
        assert_equal(series[3:7]._dates, dates[3:7])
        # Ditto
        assert_equal(series[:5]._series._data, data[:5]._data)
        assert_equal(series[:5]._series._mask, data[:5]._mask)
        assert_equal(series[:5]._dates, dates[:5])
        # With set
        series[:5] = 0
        assert_equal(series[:5]._series, [0,0,0,0,0])
        dseries = N.log(series)
        series[-5:] = dseries[-5:]
        assert_equal(series[-5:], dseries[-5:])
        # Now, using dates !
        dseries = series[series.dates[3]:series.dates[7]]
        assert_equal(dseries, series[3:7])
        
class test_functions(NumpyTestCase):
    "Some getitem tests"
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1,16)]
        dates = date_array_fromlist(dlist)
        data = masked_array(numeric.arange(15), mask=[1,0,0,0,0]*3)
        self.d = (time_series(data, dlist), data, dates)
    #
    def test_adjustendpoints(self):
        "Tests adjust_endpoints"
        (series, data, dates) = self.d
        dseries = adjust_endpoints(series, series.dates[0], series.dates[-1])
        assert_equal(dseries, series)
        dseries = adjust_endpoints(series, series.dates[3], series.dates[-3])
        assert_equal(dseries, series[3:-2])
        dseries = adjust_endpoints(series, end_date=Date('D', string='2007-01-31'))
        assert_equal(dseries.size, 31)
        assert_equal(dseries._mask, N.r_[series._mask, [1]*16])
        dseries = adjust_endpoints(series, end_date=Date('D', string='2007-01-06'))
        assert_equal(dseries.size, 6)
        assert_equal(dseries, series[:6])
        dseries = adjust_endpoints(series, 
                                   start_date=Date('D', string='2007-01-06'),
                                   end_date=Date('D', string='2007-01-31'))
        assert_equal(dseries.size, 26)
        assert_equal(dseries._mask, N.r_[series._mask[5:], [1]*16])
    #
    def test_maskperiod(self):        
        "Test mask_period"
        (series, data, dates) = self.d
        series.mask = nomask
        (start, end) = ('2007-01-06', '2007-01-12')
        mask = mask_period(series, start, end, inside=True, include_edges=True,
                           inplace=False)
        assert_equal(mask._mask, N.array([0,0,0,0,0,1,1,1,1,1,1,1,0,0,0]))
        mask = mask_period(series, start, end, inside=True, include_edges=False,
                           inplace=False)
        assert_equal(mask._mask, [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0])
        mask = mask_period(series, start, end, inside=False, include_edges=True,
                           inplace=False)
        assert_equal(mask._mask, [1,1,1,1,1,1,0,0,0,0,0,1,1,1,1])
        mask = mask_period(series, start, end, inside=False, include_edges=False,
                           inplace=False)
        assert_equal(mask._mask, [1,1,1,1,1,0,0,0,0,0,0,0,1,1,1])
    #
    def pickling(self):
        "Tests pickling/unpickling"
        (series, data, dates) = self.d
        tmp = maskedarray.loads(series.dumps())
        assert_equal(tmp._data, series._data)
        assert_equal(tmp._dates, series._dates)
        assert_equal(tmp._mask, series._mask)
        
###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()        