__all__ = ['cwt', 'ccwt', 'icwt', 'SDG', 'Morlet']

import numpy as np
from scipy.fftpack import fft, ifft

class Wavelet(object):
    '''
    Class for Wavelet object

    The Wavelet object holds the wavelet coefficients as well as information on
    how they were obtained.
    '''

    def __init__(self,wt,wavelet,weighting_function,signal_dtype,deep_copy=True):
        '''
        Initialization of Wavelet object.

        Parameters
        ----------
        wt : array
            array of wavelet coefficients
        wavelet : object
            mother wavelet object used in the creation of `wt`
        weighting_function : function
            function used in the creation of `wt`
        signal_dtype : dtype
            dtype of signal used in the creation of `wt`
        deep_copy : bool
            if true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.motherwavelet; if false, wavelet.motherwavelet will be a
            reference to the motherwavelet object.

        Returns
        -------

        Returns an instance of the Wavelet class.
        '''

        from copy import deepcopy
        self.coefs = wt[:,0:wavelet.len_signal]

        if wavelet.len_signal !=  wavelet.len_wavelet:
            self._pad_coefs = wt[:,wavelet.len_signal:]
        else:
            self._pad_coefs = None
        if deep_copy:
            self.motherwavelet = deepcopy(wavelet)
        else:
            self.motherwavelet = wavelet

        self.weighting_function = weighting_function
        self._signal_dtype = signal_dtype

    def get_gws(self):
        """
        Calculate Global Wavelet Spectrum as defined in Torrence and Compo (1998)
        """

        gws = self.get_wavelet_var()

        return gws


    def get_wes(self):
        """
        Calculate Wavelet Energy Spectrum
        """

        from scipy.integrate import trapz

        coef = 1. / (self.motherwavelet.fc * self.motherwavelet.cg)

        wes = coef * trapz(np.power(np.abs(self.coefs),2),axis=1);

        return wes

    def get_wps(self):
        """
        Calculate Wavelet Power Spectrum
        """

        wps =  1./ (self.motherwavelet.len_signal) * self.get_wes()

        return wps

    def get_wavelet_var(self):
        """
        Calculate Wavelet Variance (a.k.a. the Global Wavelet Spectrum of
        Torrence and Compo (1998))

        References

        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.
        """

        coef =  self.motherwavelet.cg * self.motherwavelet.fc

        wvar = (coef / self.motherwavelet.len_signal) * self.get_wes()

        return wvar

    def scalogram(self,show_coi=False,show_wps=False,ts = None,time = None, use_period = True, ylog_base = None,origin='top'):
        """
        Creates a simple plot of scalogram, with optional wavelet power specturm and
        time series of the transformed signal.

        Parameters
        ----------
        show_coi : bool
            set to true to see Cone of Influence
        show_wps : bool
            set to true to see the Wavelet Power Spectrum
        ts : array
            1D array containing time series data used in wavelet transform.  If set,
            time series will be plotted.
        time : array
            1D array containing time information
        use_period : bool
            set to true to see figures use period instead of scale
        ylog_base : float
            if a log scale is desired, set `ylog_base` as float. (for log 10, set
            ylog_base = 10)
        origin : 'top' or 'bottom'
            set origin of scale axis to top or bottom of figure

        Returns
        -------

        None

        Examples
        --------

        Create instance of SDG mother wavelet, normalized, using 10 scales and the
        center frequency of the Fourier transform as the characteristic frequency.
        Then, perform the continuous wavelet transform and plot the scalogram.

        x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
        data = numpy.sin(x**2)
        scales = numpy.arange(10)

        mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10), normalize = True, fc = 'center')
        wavelet = cwt(data, mother_wavelet)
        wave_coefs.scalogram(origin = 'bottom')
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from pylab import poly_between

        if ts is not None:
            show_ts = True
        else:
            show_ts = False

        if not show_wps and not show_ts:
            #only show scalogram
            figrow = 1
            figcol = 1
        elif show_wps and not show_ts:
            #show scalogram and wps
            figrow = 1
            figcol = 4
        elif not show_wps and show_ts:
            #show scalogram and ts
            figrow = 2
            figcol = 1
        else:
            #show scalogram, wps, and ts
            figrow = 2
            figcol = 4

        if time is None:
            x = np.arange(self.motherwavelet.len_signal)
        else:
            x = time

        if use_period:
            y = self.motherwavelet.scales / self.motherwavelet.fc
        else:
            y = self.motherwavelet.scales

        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(figrow,figcol,1)

        # if show wps, give 3/4 space to scalogram, 1/4 to wps
        if show_wps:
            # create temp axis at 3 or 4 col of row 1
            axt = fig.add_subplot(figrow,figcol,3)
            # get location of axtmp and ax1
            axt_pos = axt.get_position()
            ax1_pos = ax1.get_position()
            axt_points = axt_pos.get_points()
            ax1_points = ax1_pos.get_points()
            # set axt_pos left bound to that of ax1
            axt_points[0][0] = ax1_points[0][0]
            ax1.set_position(axt_pos)
            fig.delaxes(axt)

        if show_coi:
            # coi_coef is defined using the assumption that you are using
            #   period, not scale, in plotting - this handles that behavior
            if use_period:
                coi = self.motherwavelet.get_coi()
            else:
                coi = self.motherwavelet.get_coi() * self.motherwavelet.fc

            coi[coi==0]=0.00000001
            xs,ys = poly_between(np.arange(0,len(coi)),np.max(y),coi)
            ax1.fill(xs,ys,'k',alpha=0.4,zorder = 2)

        contf=ax1.contourf(x,y,np.abs(self.coefs)**2)
        fig.colorbar(contf, ax=ax1, orientation = 'vertical',format='%2.1f')

        if ylog_base is not None:
            ax1.axes.set_yscale('log', basey=ylog_base)

        if origin is 'top':
            ax1.set_ylim((y[-1],y[0]))
        elif origin is 'bottom':
            ax1.set_ylim((y[0],y[-1]))
        else:
            raise OriginError('`origin` must be set to "top" or "bottom"')

        ax1.set_xlim((x[0],x[-1]))
        ax1.set_title('scalogram')
        ax1.set_ylabel('time')
        if use_period:
            ax1.set_ylabel('period')
            ax1.set_xlabel('time')
        else:
            ax1.set_ylabel('scales')
            if time is not None:
                ax1.set_xlabel('time')
            else:
                ax1.set_xlabel('sample')

        if show_wps:
            ax2 = fig.add_subplot(figrow,figcol,4,sharey=ax1)
            if use_period:
                ax2.plot(self.get_wps(),y,'k')
            else:
                ax2.plot(self.motherwavelet.fc * self.get_wps(),y,'k')

            if ylog_base is not None:
                ax2.axes.set_yscale('log', basey=ylog_base)
            if origin is 'top':
                ax2.set_ylim((y[-1],y[0]))
            else:
                ax2.set_ylim((y[0],y[-1]))
            if use_period:
                ax2.set_ylabel('period')
            else:
                ax2.set_ylabel('scales')
            ax2.grid()
            ax2.set_title('wavelet power spectrum')

        if show_ts:
            ax3 = fig.add_subplot(figrow,2,3,sharex=ax1)
            ax3.plot(x,ts)
            ax3.plot(x,ts,'x')
            ax3.set_xlim((x[0],x[-1]))
            ax3.legend(['time series'])
            ax3.grid()
            # align time series fig with scalogram fig
            t = ax3.get_position()
            ax3pos=t.get_points()
            ax3pos[1][0]=ax1.get_position().get_points()[1][0]
            t.set_points(ax3pos)
            ax3.set_position(t)
            if (time is not None) or use_period:
                ax3.set_xlabel('time')
            else:
                ax3.set_xlabel('sample')

        plt.show()

def cwt(x,wavelet,weighting_function = lambda x: x**(-0.5), deep_copy = True):
    """
    Computes the continuous wavelet transform of x using the mother wavelet
    `wavelet`.

    This function computes the continuous wavelet transform of x using an
    instance a mother wavelet object.

    The cwt is defined as:

        T(a,b) = w(a) integral(-inf,inf)(x(t) * psi*{(t-b)/a} dt

    which is a convolution.  In this algorithm, the convolution in the time
    domain is implemented as a multiplication in the Fourier domain.

    Parameters
    ----------
    x : 1D array
        time series to be transformed by the cwt
    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family
    weighting_function:  Function used to weight
        Typically w(a) = a^(-0.5) is chosen as it ensures that the
        wavelets at every scale have the same energy.
    deep_copy : bool
        if true (default), the mother wavelet object used in the creation of
        the wavelet object will be fully copied and accessible through
        wavelet.motherwavelet; if false, wavelet.motherwavelet will be a
        reference to the motherwavelet object.

    Returns
    -------

    Returns an instance of the Wavelet class.  The coefficients of the transform
    can be obtain by the coefs() method (i.e.  wavelet.coefs() )

    Examples
    --------

    Create instance of SDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    data = numpy.sin(x**2)
    scales = numpy.arange(10)

    mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10), normalize = True, fc = 'center')
    wavelet = cwt(data, mother_wavelet)
    wave_coefs.scalogram()

    References
    ----------

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.
    """

    signal_dtype = x.dtype

    if len(x) < wavelet.len_wavelet:
        n = len(x)
        x = np.resize(x, (wavelet.len_wavelet,))
        x[n:] = 0

    # Transform the signal and motherwavelet into the Fourier domain

    xf=fft(x)
    mwf=fft(wavelet.coefs.conj(),axis=1)

    # Convolve (mult. in Fourier space)
    wt_tmp=ifft(mwf*xf[np.newaxis,:],axis=1)

    # shift output from ifft and multiply by weighting function
    wt = np.fft.fftshift(wt_tmp,axes=[1]) * weighting_function(wavelet.scales[:,np.newaxis])

    # if motherwavelet and signal are real, only keep real part of transform
    wt=wt.astype(np.lib.common_type(wavelet.coefs,x))

    return Wavelet(wt,wavelet,weighting_function,signal_dtype,deep_copy)

def ccwt(x1,x2,wavelet):
    '''
    Compute the continuous cross-wavelet transform of 'x1' and 'x2' using the
    mother wavelet 'wavelet', which is an instance of the MotherWavelet class.

    Parameters
    ----------

    x1,x2 : 1D array
        time series used to compute cross-wavelet transform
    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family

    Returns
    -------

    Returns an instance of the Wavelet class.
    '''

    xwt=cwt(x1,wavelet)*np.conjugate(cwt(x2,wavelet))

    return xwt

def icwt(wavelet):
    """
    Compute the inverse continuous wavelet transform.

    Parameters
    ----------
    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family

    References
    ----------

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.
    """
    from scipy.integrate import trapz

    # if origional wavelet was created using padding, make sure to include
    #   information that is missing after truncation (see self.coefs under __init__
    #   in class Wavelet.

    if wavelet.motherwavelet.len_signal !=  wavelet.motherwavelet.len_wavelet:
        full_wc = np.c_[wavelet.coefs,wavelet._pad_coefs]
    else:
        full_wc = wavelet.coefs
    # get unweighted wavelet coefficents and take fft
    wc = full_wc * wavelet.weighting_function(wavelet.motherwavelet.scales[:,np.newaxis])
    wcf = fft(wc,axis=1)
    # get unweighted motherwavelet coeffifientts and take fft
    mw = wavelet.motherwavelet.coefs / (wavelet.motherwavelet.scales[:,np.newaxis]**2)
    mwf = fft(mw,axis=1)
    # perform inverse continuous wavelet transform and make sure the result is the same type
    #  (real or complex) as the origional data used in the transform
    x=((1 / wavelet.motherwavelet.cg) * trapz(np.fft.fftshift(ifft(
        wcf * mwf,axis=1),axes=[1]),axis=0)).astype(wavelet._signal_dtype)

    return x[0:wavelet.motherwavelet.len_signal]

class MotherWavelet(object):
    """
    Class for MotherWavelets

    Contains methods related to mother wavelets.  Also used to ensure that new
    mother wavelet objects contain the minimum requirements to be used in the
    cwt related functions.
    """

    @staticmethod
    def get_coefs(self):
        """
        raise error method for calculating mother wavelet coefficients is
        missing!  To follow the convention in the literature, please define your
        COI coef as a function of period, not scale - this will ensure
        compatibility with the scalogram method.
        """

        raise NotImplementedError('get_coefs needs to be implemented for the mother wavelet')

    @staticmethod
    def get_coi_coef(sampf):
        """
        raise error if Cone of Influence coefficient is not set in subclass wavelet
        """

        raise NotImplementedError('coi_coef needs to be implemented in subclass wavelet')

    #add methods for computing cone of influence and mask
    def get_coi(self):
        """
        Compute cone of influence
        """

        y1 =  self.coi_coef*np.arange(0,self.len_signal/2)
        y2 = -self.coi_coef*np.arange(0,self.len_signal/2)+y1[-1]
        coi = np.r_[y1,y2]
        self.coi = coi
        return coi

    def get_mask(self):
        """
        get mask for cone of influence.

        Sets self.mask as an array of bools for use in np.ma.array('',mask=mask)
        """

        mask = np.ones(self.coefs.shape)
        masks = self.coi_coef*self.scales
        for s in range(0,len(self.scales)):
            if (s != 0) and (int(np.ceil(masks[s])) < mask.shape[1]):
                mask[s,np.ceil(int(masks[s])):-np.ceil(int(masks[s]))]=0
        self.mask = mask.astype(bool)
        return self.mask


class SDG(MotherWavelet):
    """
    SDG(self, len_signal = None, pad_to = None, scales = None, sampf = 1,
        normalize = True, fc = 'bandpass')

    Class for the SDG MotherWavelet (a subclass of MotherWavelet).

    Parameters
    ----------

    len_signal : int
        length of time series to be decomposed
    pad_to : int
        pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set).
    scales : array
        array of scales used to initialize the mother wavelet
    sampf : float
        sample frequency of the time series to be decomposed
    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy)
    fc : string
        Characteristic frequency - use the 'bandpass' or 'center' frequency of
        the Fourier spectrum of the mother wavelet to relate scale to period
        (default is 'bandpass')

    Returns
    -------

    Returns an instance of the MotherWavelet class which is used in the cwt and
    icwt functions.

    Examples
    --------

    Create instance of SDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    data = numpy.sin(x**2)
    scales = numpy.arange(10)

    mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10),normalize = True, fc = 'center')
    wavelet = cwt(data, mother_wavelet)
    wave_coefs.scalogram()

    Notes
    -----

    The coefficient that is used to defined the 'Cone of Influence' is based on
    the use of period, not scale - if you wish to use scale, multiply coi_coef()
    by `motherwavelet.fc`.

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.
    """

    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, fc = 'bandpass'):
        self.name='second degree of a Gaussian (mexican hat)'
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = normalize
        # coi_coef defined under the assumption that period is used, not scale
        self.coi_coef = 2*np.pi*np.sqrt(2./5.)/self.sampf ;#Torrence and Compo 1998

        #set total length of wavelet to account for zero padding
        if pad_to is None:
            self.len_wavelet = len_signal
        else:
            self.len_wavelet = pad_to

        #set admissibility constant
        if normalize:
            self.cg = 4 * np.sqrt(np.pi) / 3
        else:
            self.cg = np.pi

        #define characteristic frequency
        if fc is 'bandpass':
            self.fc = np.sqrt(5./2.)/(2*np.pi)
        elif fc is 'center':
            self.fc = np.sqrt(2.)/(2*np.pi)
        else:
            raise CharacteristicFrequencyError("fc = %s not defined"%(fc,))

        #compute coefficients for the dilated mother wavelet

        self.coefs = self.get_coefs()

    def get_coefs(self):
        """
        Calculate the coefficients for the mother wavelet SDG
        """

        #Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self.len_wavelet/2.,self.len_wavelet/2.)

        #find motherwavelet coefficients at each scale

        xsd = -xi * xi / (self.scales[:,np.newaxis] * self.scales[:,np.newaxis])

        if self.normalize is True:
            c=2./(np.sqrt(3)*np.power(np.pi,0.25))
        else:
            c=1.

        mw = c * (1. + xsd) * np.exp(xsd / 2.)

        self.coefs = mw

        return mw

class Morlet(MotherWavelet):
    """
    Morlet(self, len_signal = None, pad_to = None, scales = None,
           sampf = 1, f0 = 6)

    Class for the SDG MotherWavelet (a subclass of MotherWavelet).

    Parameters
    ----------

    len_signal : int
        length of time series to be decomposed
    pad_to : int
        pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set).
    scales : array
        array of scales used to initialize the mother wavelet
    sampf : float
        sample frequency of the time series to be decomposed
    f0 : float
        central frequency of the Morlet mother wavelet.  The Fourier spectrum of
        the Morlet wavelet appears as a Gaussian centered on f0.  f0 defaults
        to a value of 0.849 (the angular frequency would be ~5.336)

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt
    and icwt functions.

    Examples
    --------

    Create instance of Morlet mother wavelet using 10 scales, perform the
    continuous wavelet transform, and plot the resulting scalogram.

    x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    data = numpy.sin(x**2)
    scales = numpy.arange(10)

    mother_wavelet = Morlet(len_signal=len(data), scales = np.arange(10))
    wavelet = cwt(data, mother_wavelet)
    wave_coefs.scalogram()

    Notes
    -----

    * Morlet wavelet is defined as having unit energy, so the `normalize` flag
      will always be set to True

    * The Morlet wavelet will always use f0 as it's characteristic frequency, so
      fc is set as f0

    * The coefficient that is used to defined the 'Cone of Influence' is based
      on the use of period, not scale - if you wish to use scale, multiply
      coi_coef() by `motherwavelet.fc`.


    References
    ----------

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, f0 = 0.849):
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = True
        # coi_coef defined under the assumption that period is used, not scale
        ###FIX###
        self.coi_coef = ((4*np.pi)/(f0 + np.sqrt(2. + f0**2)))/np.sqrt(2)/self.sampf ;#Torrence and Compo 1998
        self.name='Morlet'

        #set total length of wavelet to account for zero padding
        if pad_to is None:
            self.len_wavelet = len_signal
        else:
            self.len_wavelet = pad_to

        #define characteristic frequency
        self.fc = f0

        #set admissibility constant
        ###FIX - based on http://www.ecs.syr.edu/Faculty/lewalle/tutor/node14.html
        self.cg = 1.44057

        #compute coefficients for the dilated mother wavelet

        self.coefs = self.get_coefs()

    def get_coefs(self):
        """
        Calculate the coefficients for the mother wavelet SDG
        """

        #Create array containing values used to evaluate the wavelet function

        xi=np.arange(-self.len_wavelet/2.,self.len_wavelet/2.)

        #find motherwavelet coefficients at each scale

        xsd = xi / (self.scales[:,np.newaxis])

        mw = np.power(np.pi,-0.25) * \
                     (np.exp(np.complex(1j) * 2. * np.pi * self.fc * xsd) - \
                     np.exp(-np.power((2. * np.pi * self.fc),2) / 2.)) *  \
                     np.exp(-np.power(xsd,2) / 2.)

        self.coefs = mw

        return mw
