__all__ = ['cwt', 'ccwt', 'icwt', 'SDG', 'Morlet']

import numpy as np
from scipy.fftpack import fft, ifft

class Wavelet(object):
    def __init__(self,wt,wavelet,weighting_function,signal_dtype):
        self.coefs = wt[:,0:wavelet.len_signal]

        if wavelet.len_signal !=  wavelet.len_wavelet:
            self._pad_coefs = wt[:,wavelet.len_signal:]
        else:
            self._pad_coefs = None

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

        Notes
        -----

        References

        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavlet Analysis.
          Bulletin of the American Meteorological Society, 79, 1, pp. 61-78.

        """

        coef =  self.motherwavelet.cg * self.motherwavelet.fc

        wvar = (coef / self.motherwavelet.len_signal) * self.get_wes()

        return wvar

    def scalogram(self,show_wps=False,ts = None,time = None, use_period = True, ylog_base = None,origin='top'):
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
            figcol = 2
        elif not show_wps and show_ts:
            #show scalogram and ts
            figrow = 2
            figcol = 1
        else:
            #show scalogram, wps, and ts
            figrow = 2
            figcol = 2

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

        contf=ax1.contourf(x,y,np.abs(self.coefs)**2)
        fig.colorbar(contf, ax=ax1, orientation = 'vertical',format='%2.1f')

        if ylog_base is not None:
            ax1.axes.set_yscale('log', basey=ylog_base)

        if origin is 'top':
            ax1.set_ylim((y[-1],y[0]))
        else:
            ax1.set_ylim((y[0],y[-1]))

        ax1.set_xlim((x[0],x[-1]))
        ax1.set_title('scalogram')
        ax1.set_ylabel('time')
        if use_period:
            ax1.set_ylabel('period')
        else:
            ax1.set_ylabel('scales')

        if show_wps:
            ax2 = fig.add_subplot(figrow,figcol,2)
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
            ax3 = fig.add_subplot(figrow,1,2)
            ax3.plot(x,ts)
            ax3.plot(x,ts,'x')
            ax3.set_xlim((x[0],x[-1]))
            ax3.legend(['time series'])
            ax3.set_xlabel('time')
            ax3.grid()

        plt.show()

def cwt(x,wavelet,weighting_function = lambda x: x**(-0.5)):
    """
    Computes the continuous wavelet transform of x using the mother wavelet
    `wavelet`.

    This function computes the continuous wavelet transform of x using an
    instance a mother wavelet object.

    The cwt is defined as:

        T(a,b) = w(a) integral(-inf,inf)(x(t) * psi*{(t-b)/a} dt

    which is a convolution.  In this algorithm, the convolution in the time
    domain is implemented as a multiplicaion in the Fourier domain.

    Parameters
    ----------
    x : 1D array
        time series to be transformed by the cwt

    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family

    weighting_function:  Function used to weight
        Typically w(a) = a^(-0.5) is choosen as it ensures that the
        wavelets at every scale have the same energy.

    Returns
    -------

    Returns an instance of the Wavelet class.  The coefficients of the transform
    can be obtain by the coefs() method (i.e.  wavelet.coefs() )

    Examples
    --------

    Create instance of SDG mother wavelet, nomalized, using 10 scales and the
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

    return Wavelet(wt,wavelet,weighting_function,signal_dtype)

def ccwt(x1,x2,wavelet):
    '''
    Compute the continuous cross-wavelet transform of 'x1' and 'x2' using the
    mother wavelet 'wavelet', which is an instance of the motherwavelet class.

    Parameters
    ----------
    x1,x2 : 1D array
        time series used to compute cross-wavelet transform

    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family

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

    Notes
    -----

    References

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

    """
    @staticmethod
    def get_coefs(self):
        """
        raise error method for calculating mother wavelet coefficients is
        missing!
        """
        raise NotImplementedError('get_coefs needs to be implemented for the mother wavelet')

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
        array of scales used to initilize the mother wavelet
    sampf : float
        sample frequency of the time series to be decomposed
    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy)
    fc : string
        Characteristic frequency - use the 'bandpass' or 'center' frequency of
        the Fourier specturm of the mother wavelet to relate scale to period
        (default is 'bandpass')

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt and
    icwt functions.

    Examples
    --------

    Create instance of SDG mother wavelet, nomalized, using 10 scales and the
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

    References

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """
    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, fc = 'bandpass'):
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = normalize

        self.name='second degree of a gaussian (mexican hat)'

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
        array of scales used to initilize the mother wavelet
    sampf : float
        sample frequency of the time series to be decomposed
    f0 : float
        central frequency of the Morlet mother wavelet.  The Fourier specturm of
        the Morlet wavelet appears as a gaussian centered on f0.  f0 defaults
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

    Morlet wavelet is defined as having unit energy, so the `normalize` flag
    will always be set to True

    The Morlet wavelet will always use f0 as it's characterstic frequency, so fc
    is set as f0

    References

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """
    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, f0 = 0.849):
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = True

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
