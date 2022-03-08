""" Feature extraction module"""
from numpy import sum, log2, argmax, trapz, gradient, sqrt, var
from scipy.signal import welch 
from scipy.stats import skew, kurtosis

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

def Hjorth(t, X):
    """Computes the variance, Hjorth mobility and Hjorth complexity of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: V (float), Variance of the timeseries;
             HM (float), Hjorth Mobility of the timeseries;
             HC (float), Hjorth Complexity of the timeseries.
    """

    Xp = gradient(X, t[1]-t[0])
    Xpp = gradient(Xp, t[1]-t[0])

    V = var(X) # Variance 
    Vp = var(Xp)
    Vpp = var(Xpp)

    HM = sqrt(Vp/V) # Hjorth Mobility
    HMp = sqrt(Vpp/Vp)

    HC =  HMp/HM # Hjorth Complexity 

    return V, HM, HC

def Skew(t, X):
    """Computes the skewness of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: SK (float), Skewness of the timeseries.
    """

    SK = skew(X)

    return SK

def Kurt(t, X):
    """Computes the kurtosis of the signal.

    Intent(in): Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: KT (float), Kurtosis of the timeseries.
    """

    KT = kurtosis(X)

    return KT