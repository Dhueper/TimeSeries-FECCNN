import os
import sys

from matplotlib import pyplot as plt
from numpy import zeros, ones, mean, asfortranarray, linspace, pi, amax, array, append
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import fortran_ts

def reshape_2pow(t, X):
    N = len(X)
    #Reshape to fit a power of 2. 
    for k in range(0, 1000):
        if 2**k > N:
            break
    t_k = linspace(0,amax(t),2**(k-1)) 
    f = interp1d(t, X, fill_value='extrapolate')
    X_k = f(t_k)
    return [t_k, X_k]  

def spectral_derivative(t, X):
    N = len(X)
    delta_t = t[1]-t[0]

    #Spectral domain
    xf = fft(X, N)
    f = fftfreq(N,delta_t)  

    dx_spec = ifft(2*pi*f*1j*xf, N).real 

    return dx_spec/amax(abs(dx_spec))

def rectangular_transform(X, dx_spec, alpha, beta):
    """Transforms a time series into another formed by rectangular signals.

    Input: X (numpy.array), time series;
           dx_spec (numpy.array), spectral derivative of the time series;
           alpha (float), threshold used for discontinuity detection (alpha<1);
           beta (float), maximum length allowed in each pulse of the rectangular signal.

    Returns: X_rect (numpy.array), transformed time series.
    """
    N = len(X)
    X_rect = array([])
    X_aux = []  
    for i in range(0, N):
        X_aux.append(X[i])
        #Looking for discontinuities 
        if abs(dx_spec[i]) > alpha and len(X_aux)>int(N*beta):
            X_rect = append(X_rect, ones(len(X_aux))*mean(array(X_aux)))
            X_aux = []
    if len(X_aux)>0:
         X_rect = append(X_rect, ones(len(X_aux))*mean(array(X_aux)))

    return X_rect

def phi(t, m, n):
    """Base Haar functions.

    Input: t (float), timestamp;
           m (integer), order of the function;
           n (integer), coeffcieint (n=0, 1, ..., 2**n).

    Returns: phi (float), Haar function value.
    """
    def psi(t):
        if 0 <= t and t < 0.5:
            return 1
        elif 0.5 <= t and t < 1:
            return -1
        else:
            return 0

    return 2**(-m) * psi(2**(-m) * t - n)

def haar_coef(t, X, order):
    N = len(t)
    for k in range(0, 1000):
        if 2**k > N:
            break
    t_k = linspace(0,24,2**(k-1))
    f = interp1d(t, X, fill_value='extrapolate')
    X_k = f(t_k)
    
    return [t_k, X_k] 

if __name__ == "__main__":
    name = 'W_Lights'
    [t0, X] = test_function.read('data/Sanse/20220301.plt', name)
    [t, Y] = reshape_2pow(t0, X)  
    #Mean value filter 
    for _ in range(0,50):
        Y = fortran_ts.time_series.mvf(asfortranarray(Y), 1)
        # Y[0] = 2*Y[1] - Y[2] 
        # Y[len(Y)-1] = 2*Y[len(Y)-2] - Y[len(Y)-3]


    # dx_spec = spectral_derivative(t, X)
    dy_spec = spectral_derivative(t, Y)

    plt.figure()
    # plt.plot(t, dx_spec, 'b')
    plt.plot(t, dy_spec, 'r')
    plt.xlabel('t [h]')
    plt.ylabel('dx_spec')
    plt.title('Spectral derivative')
    # plt.show()

    alpha = 0.15 #Discontinuity threshold 
    beta = 1./400 #Length threshold 
    Y_rect = rectangular_transform(Y, dy_spec, alpha, beta)

    # X_k  = haar_coef(t, Y_rect, 2)

    plt.figure()
    plt.plot(t0, X, 'g')
    plt.plot(t, Y, 'b')
    plt.plot(t, Y_rect, 'r')
    # plt.plot(t_rect, X_k)
    plt.xlabel('t [h]')
    plt.ylabel('P [W]')
    plt.title('Power consumption')
    plt.show()


   
