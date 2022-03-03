""" Feature extraction module"""
from numpy import sum
from scipy.signal import welch 

def SPow(t,X):
    """Computes the Spectral Power of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SP (float), Spectral Power of the timeseries.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample, detrend=False)

    N = len(Pxx)
    SP = sum(Pxx)/N

    return SP