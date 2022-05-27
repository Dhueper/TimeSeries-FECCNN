"""Rectangular signal module"""
import os
import sys

from matplotlib import pyplot as plt
from numpy import zeros, ones, mean, asfortranarray, linspace, pi, amax, array, append, trapz, dot, log2
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
from mpmath import quadgl

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import fortran_ts

def reshape_2pow(t, X):
    """Reshape to fit a power of 2.

    Input: t (numpy.array), timestamps;
           X (numpy.array), time series.

    Returns: [t_k, X_k] (numpy.array), reshaped series.
    """
    N = len(X)
    for k in range(0, 1000):
        if 2**k > N:
            break
    t_k = linspace(0,amax(t),2**(k)) 
    f = interp1d(t, X, fill_value='extrapolate')
    X_k = f(t_k)
    return [t_k, X_k]  

def spectral_derivative(t, X):
    """Computes the normalized spectral derivative of a time series.

    Intent(in): t (numpy.array), timestamps;
                X (numpy.array), time series.

    Returns: dx_spec_norm (numpy.array), normalized spectral derivative back in the time domain.
    """
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
        if abs(dx_spec[i]) > alpha and len(X_aux)>=int(beta*N):
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
        elif 0.5 <= t and t <= 1:
            return -1
        else:
            return 0

    return 2**(m/2.) * psi(2**(m) * t - n)

def haar_coef(t, X, order):
    """Extracts the coeffcients of the Haar series expansion.

    Input: t (numpy.array), timestamps;
           X (numpy.array), time series;
           order (int), maximum order of the coefficients.

    Returns: c_haar (list), Haar series coefficients.
    """
    t1 = t/amax(t)
    N = len(t1)
    haar_func = zeros(N)
    c_haar = [] 
    for m in range(0, order):
        c_haar.append([])
        for n in range(0, 2**m):
            for i in range(0, N):
                haar_func[i] = phi(t1[i], m, n) 
            # c_haar[m].append(trapz(X*haar_func, t1))
            c_haar[m].append(dot(X,haar_func)/N)

    return c_haar


if __name__ == "__main__":
    # name = 'W_Lights'
    # name = 'W_Computers'
    name = 'W_Gas_boiler'
    [t0, X] = test_function.read('data/Sanse/20220301.plt', name) 
    t0 = t0 / amax(t0)
    Z = zeros(len(X))
    Z[:] = X[:]  
    #Mean value filter 
    for _ in range(0,50):
        Z = fortran_ts.time_series.mvf(asfortranarray(Z), 0)
        # Z[0] = 2*Z[1] - Z[2] 
        # Z[len(Z)-1] = 2*Z[len(Z)-2] - Z[len(Z)-3]

    #Reshape to fit a power of 2. 
    [t, Y] = reshape_2pow(t0, Z) 

    # dx_spec = spectral_derivative(t, X)
    dy_spec = spectral_derivative(t, Y)

    plt.figure()
    # plt.plot(t, dx_spec, 'b')
    plt.plot(t, dy_spec, 'r')
    plt.xlabel('t [h]')
    plt.ylabel('dx_spec')
    plt.title('Spectral derivative')
    # plt.show()

    alpha = 0.05 #Discontinuity threshold 
    beta = 2**(-6) #Length threshold 
    Y_rect = rectangular_transform(Y, dy_spec, alpha, beta)

    #Haar series expansion
    order =  int(log2(beta**(-1))) 
    # order = 7
    c_haar = haar_coef(t, Y, order)
    N = len(t)
    c = zeros(N)
    Y_haar = ones(N)
    Y_haar = Y_haar * mean(Y) 

    for m in range(0, order):
        for n in range(0, 2**m):
            for i in range(0, N):
                c[i] = phi(t[i], m, n) * c_haar[m][n]  
            Y_haar  = Y_haar + c 
    # Y_haar = Y_haar / 2**(order-2)

    plt.figure()
    plt.plot(t0, X, 'g')
    plt.plot(t, Y, 'c')
    plt.plot(t, Y_rect, 'r')
    plt.plot(t, Y_haar, 'b')
    plt.xlabel('t [h]')
    plt.ylabel('P [W]')
    plt.title('Power consumption')
    plt.show()

    N_coef = sum(array([2**i for i in range(0,order)])) + 1
    print('Rectangular signal:', len(Y_rect), 'points')
    print('Haar signal:', N_coef, 'coefficients')




    # t_test = linspace(0, 1, 4)
    # X_test = array([9., 7., 3., 5.])
    # order = 4
    # c_haar = haar_coef(t_test, X_test, order)
    # print(c_haar)
    # N = len(X_test)
    # Y_test = ones(N)
    # c = zeros(N)
    # c0 = zeros(N)
    # c_phi0 = zeros(N)
    # Y_test = Y_test * mean(X_test) 

    # for m in range(0, order):
    #     for n in range(0, 2**m):
    #         for i in range(0, N):
    #             c[i] = phi(t_test[i], m, n) * c_haar[m][n]  
    #         Y_test  = Y_test + c 
    # Y_test = Y_test / 2**(order-2)

    # print(Y_test)



   
