""" Feature extraction module"""
from numpy import sum, log2, argmax, trapz, gradient, sqrt, var
from scipy.signal import welch 
from scipy.stats import skew, kurtosis

class Features():
    def __init__(self, t, X):
        """Computes different features of the signal.

        Intent(in): Intent(in): t (numpy.array), timestamps;
                    X (numpy.array), time series.
        """
        self.t = t
        self.X = X

        self.f_sample = 1./(t[1]-t[0])
        self.f, self.Pxx = welch(self.X, self.f_sample)
        self.N = len(self.Pxx)

        self.Xp = gradient(self.X, self.t[1]-self.t[0])

        self.fdict = {} 

    def SPow(self):
        """Computes the Spectral Power of the signal.

        Intent(in): self (object), class Features.

        Returns: SPW (float), Spectral Power of the timeseries.
        """

        SPW = sum(self.Pxx)/self.N
        self.fdict['SPW'] = SPW 

        return SPW

    def SEnt(self):
        """Computes the Spectral Entropy of the signal.

        Intent(in): self (object), class Features.

        Returns: SE (float), Spectral Entropy of the timeseries.
        """

        SE = - sum(self.Pxx * log2(self.Pxx + 1e-8)) / log2(self.N)
        self.fdict['SE'] = SE

        return SE

    def SPeak(self):
        """Computes the Spectral Peak of the signal and its asociated frequency.

        Intent(in): self (object), class Features.

        Returns: SP (float), Spectral Peak of the timeseries;
                fP (float), Peak frequency.
        """

        SP = max(self.Pxx)
        fP = self.f[argmax(self.Pxx)] 

        self.fdict['SP'] = SP
        self.fdict['fP'] = fP

        return SP, fP

    def SCen(self):
        """Computes the Spectral Centroid of the signal and its asociated frequency.

        Intent(in): self (object), class Features.

        Returns: SC (float), Spectral Centroid of the timeseries.
        """

        SC = sum(self.f * self.Pxx) / sum(self.Pxx)
        self.fdict['SC'] = SC

        return SC

    def BW(self):
        """Computes the AM and FM bandwidth of the signal.

        Intent(in): self (object), class Features.

        Returns: AM (float), AM bandwidth of the timeseries;
                FM (float), FM bandwidth of the timeseries.
        """

        E = trapz(self.Pxx, x=self.f)

        AM = sqrt(trapz(self.Xp**2., x=self.t) / E)

        omega = trapz(self.Xp * self.X**2., x=self.t) / E

        FM = sqrt(trapz((self.Xp - omega)**2. * self.X**2., x=self.t) / E)

        self.fdict['AM'] = AM
        self.fdict['FM'] = FM 

        return AM, FM

    def Hjorth(self):
        """Computes the variance, Hjorth mobility and Hjorth complexity of the signal.

        Intent(in): self (object), class Features.

        Returns: V (float), Variance of the timeseries;
                HM (float), Hjorth Mobility of the timeseries;
                HC (float), Hjorth Complexity of the timeseries.
        """

        Xpp = gradient(self.Xp, self.t[1]-self.t[0])

        V = var(self.X) # Variance 
        Vp = var(self.Xp)
        Vpp = var(Xpp)

        HM = sqrt(Vp/V) # Hjorth Mobility
        HMp = sqrt(Vpp/Vp)

        HC =  HMp/HM # Hjorth Complexity 

        self.fdict['Var'] = V 
        self.fdict['HM'] = HM 
        self.fdict['HC'] = HC 

        return V, HM, HC

    def Skew(self):
        """Computes the skewness of the signal.

        Intent(in): self (object), class Features.

        Returns: SK (float), Skewness of the timeseries.
        """

        SK = skew(self.X)
        self.fdict['SK'] = SK 

        return SK

    def Kurt(self):
        """Computes the kurtosis of the signal.

        Intent(in): self (object), class Features.

        Returns: KT (float), Kurtosis of the timeseries.
        """

        KT = kurtosis(self.X)
        self.fdict['KT'] = KT 

        return KT