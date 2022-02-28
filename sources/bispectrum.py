from numpy import array, linspace

from stingray import lightcurve
from stingray.bispectrum import Bispectrum

def bispectral_transform(t, X):
    """Computes the High Order Spectral Transform based on the
    Fourier transform of third order auto-correlation function.

    Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: bs (object), bispectrum of the time series.
    """

    lc = lightcurve.Lightcurve(t,X, dt=t[1]-t[0])

    bs = Bispectrum(lc, maxlag=10, window="uniform", scale="biased")

    return bs