""" Feature extraction module"""
from numpy import sum, log2, argmax, trapz, gradient, sqrt
from scipy.signal import welch 

def SPow(t,X):
    """Computes the Spectral Power of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SP (float), Spectral Power of the timeseries.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample)

    N = len(Pxx)
    SP = sum(Pxx)/N

    return SP

def SEnt(t,X):
    """Computes the Spectral Entropy of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SE (float), Spectral Entropy of the timeseries.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample)

    N = len(Pxx)
    SE = - sum(Pxx * log2(Pxx + 1e-8)) / log2(N)

    return SE

def SPeak(t,X):
    """Computes the Spectral Peak of the signal and its asociated frequency.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SP (float), Spectral Peak of the timeseries;
             fP (float), Peak frequency.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample)

    SP = max(Pxx)
    fP = f[argmax(Pxx)] 

    return SP, fP

def SCen(t,X):
    """Computes the Spectral Centroid of the signal and its asociated frequency.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SC (float), Spectral Centroid of the timeseries.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample)

    SC = sum(f * Pxx) / sum(Pxx)

    return SC

def BW(t, X):
    """Computes the AM and FM bandwidth of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: AM (float), AM bandwidth of the timeseries;
             FM (float), FM bandwidth of the timeseries.
    """
    f_sample = 1./(t[1]-t[0])
    f, Pxx = welch(X, f_sample)

    E = trapz(Pxx, x=f)

    Xp = gradient(X, t[1]-t[0])

    AM = sqrt(trapz(Xp**2., x=t) / E)

    omega = trapz(Xp * X**2., x=t) / E

    FM = sqrt(trapz((Xp - omega)**2. * X**2., x=t) / E)

    return AM, FM