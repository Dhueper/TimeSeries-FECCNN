import os
import sys

from matplotlib import pyplot as plt
from numpy import zeros, mean, asfortranarray, linspace, amax, ones, sum, array, transpose
from numpy.linalg import lstsq
from numpy import random
from scipy.interpolate import interp1d

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum
import clustering
import feature_extraction
import rectangular_signal
import haar
import fortran_ts

#%%Bispectrum 
def bispectrum_example():
    #Square function transformation 
    [t, X] = test_function.sinusoidal_function()

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Sinusoidal function') 

    bs = bispectrum.bispectral_transform(t, X)

    plt.figure()
    bs.plot_mag()
    plt.show()


#%% VMD clustering 
def VMD_example():
    #Sinusoidal example 
    [t, X] = test_function.sinusoidal_function()

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Sinusoidal function') 

    VMD_modes = 3
    u = clustering.VMD_clustering(t, X, VMD_modes)

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('VMD') 
    for i in range(0, VMD_modes):
        plt.subplot(VMD_modes, 1, i+1)
        plt.plot(t, u[i, :])

    plt.show()


    #Square + triangle functions 
    [t, X] = test_function.square_function3()
    [t, Y] = test_function.square_function2()

    plt.figure()
    plt.plot(t,X+Y)
    plt.xlabel('t')
    plt.ylabel('X+Y (t)')
    plt.title('Square + triangle function') 

    VMD_modes = 2
    u = clustering.VMD_clustering(t, X+Y, VMD_modes)

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('VMD') 
    for i in range(0, VMD_modes):
        plt.subplot(VMD_modes, 1, i+1)
        plt.plot(t, u[i, :])

    plt.show()

#%% EWT clustering 
def EWT_example():
    #Sinusoidal example 
    [t, X] = test_function.sinusoidal_function()

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Sinusoidal function') 

    EWT_modes = 4
    ewt = clustering.EWT_clustering(X, EWT_modes)

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('EWT') 
    for i in range(0, EWT_modes):
        plt.subplot(EWT_modes, 1, i+1)
        plt.plot(t, ewt[:, i])

    plt.show()


    #Square + triangle functions 
    [t, X] = test_function.square_function3()
    [t, Y] = test_function.square_function2()

    plt.figure()
    plt.plot(t,X+Y)
    plt.xlabel('t')
    plt.ylabel('X+Y (t)')
    plt.title('Square + triangle function') 

    EWT_modes = 3
    ewt = clustering.EWT_clustering(X+Y, EWT_modes)

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('EWT') 
    for i in range(0, EWT_modes):
        plt.subplot(EWT_modes, 1, i+1)
        plt.plot(t, ewt[:, i])

    plt.show()

#%% EWT clustering 
def EMD_example():
    #Sinusoidal example 
    [t, X] = test_function.sinusoidal_function()

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Sinusoidal function') 

    max_IMF = 3
    IMF = clustering.EMD_clustering(t, X, max_IMF)
    EMD_modes = IMF.shape[0]

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('EMD') 
    for i in range(0, EMD_modes):
        plt.subplot(EMD_modes, 1, i+1)
        plt.plot(t, IMF[i, :])

    plt.show()


    #Square + triangle functions 
    [t, X] = test_function.square_function3()
    [t, Y] = test_function.square_function2()

    plt.figure()
    plt.plot(t,X+Y)
    plt.xlabel('t')
    plt.ylabel('X+Y (t)')
    plt.title('Square + triangle function') 

    max_IMF = 2
    IMF = clustering.EMD_clustering(t, X, max_IMF)
    EMD_modes = IMF.shape[0]

    plt.figure()
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('EMD') 
    for i in range(0, EMD_modes):
        plt.subplot(EMD_modes, 1, i+1)
        plt.plot(t, IMF[i, :])

    plt.show()

def feature_extraction_example():
    [t, X] = test_function.sinusoidal_function()

    Features = feature_extraction.Features(t, X)

    Mean = Features.Mean()
    print("Mean: M=", Mean)

    Max, Min = Features.Max_Min()
    print("Maximum: Max=", Max)
    print("Minimum: Min=", Min)

    SPW = Features.SPow()
    print("Spectral Power: SPW=", SPW)

    SE = Features.SEnt()
    print("Spectral Entropy: SE=", SE)

    SP, fP = Features.SPeak()
    print("Spectral Peak: SP=", SP)
    print("Peak frequency: fP=", fP)

    SC = Features.SCen()
    print("Spectral Centroid: SC=", SC)

    AM, FM, E = Features.BW()
    print("AM bandwidth: AM=", AM)
    print("FM bandwidth: FM=", FM)
    print("Energy: E=", E)

    V, HM, HC = Features.Hjorth()
    print("Variance: Var=", V)
    print("Hjorth Mobility: HM=", HM)
    print("Hjorth Complexity: HC=", HC)

    SK = Features.Skew()
    print("Skewness: SK=", SK)

    KT = Features.Kurt()
    print("Kurtosis: KT=", KT)

    # print(Features.fdict)


if __name__ == "__main__":
    #Run examples 
    # bispectrum_example()

    # VMD_example()

    # EWT_example()

    # EMD_example()

    feature_extraction_example()

    # signal = [] 
    # signal_coef = [] 
    # name_list = {'W_Computers':4, 'W_Lights':4, 'W_Gas_boiler':4} 
    # ct = 0
    # for name in name_list.keys():
    #     [t0, X] = test_function.read('data/Sanse/20220301.plt', name) 
    #     t0 = t0 / amax(t0)
    #     Z = zeros(len(X))
    #     Z[:] = X[:]  
    #     #Mean value filter 
    #     for _ in range(0,50):
    #         Z = fortran_ts.time_series.mvf(asfortranarray(Z), 0)

    #     #Reshape to fit a power of 2. 
    #     [t, Y] = rectangular_signal.reshape_2pow(t0, Z) 

    #     #Haar series expansion
    #     order = name_list[name] 
    #     c_haar = rectangular_signal.haar_coef(t, Y, order)
    #     N = len(t)
    #     c = zeros(N)
    #     Y_haar = ones(N)
    #     Y_haar = Y_haar * mean(Y) 

    #     signal_coef.append([])
    #     signal_coef[ct].append(mean(Y))

    #     for m in range(0, order):
    #         for n in range(0, 2**m):
    #             for i in range(0, N):
    #                 c[i] = rectangular_signal.phi(t[i], m, n) * c_haar[m][n]  
    #             signal_coef[ct].append(c_haar[m][n]) 
    #             Y_haar  = Y_haar + c 

    #     N_coef = sum(array([2**i for i in range(0,order)])) + 1
    #     print('Reshaped signal:', len(Y), 'points')
    #     print('Haar signal:', N_coef, 'coefficients')
    #     noise = random.normal(0, 0.1*amax(Y_haar), len(Y_haar))
    #     Y_haar = Y_haar + noise
    #     signal.append(Y_haar)
    #     ct += 1

    #     plt.figure()
    #     plt.plot(t0, X, 'g')
    #     plt.plot(t, Y, 'c')
    #     plt.plot(t, Y_haar, 'b')
    #     plt.xlabel('t [h]')
    #     plt.ylabel('P [W]')
    #     plt.title('Power consumption' + name)
    #     # plt.show()

    # signal_coef = transpose(array(signal_coef))
    # print(type(signal_coef), signal_coef.shape)

    # general = 0.4*signal[0] + 0.6*signal [1] + signal[2] 
    # c_haar = rectangular_signal.haar_coef(t, general, order)
    # general_coef = [mean(general)]
    # for m in range(0, order):
    #     for n in range(0, 2**m):
    #           general_coef.append(c_haar[m][n])
    # general_coef = transpose(array(general_coef))
    # print(type(general_coef), general_coef.shape)

    # #Solve linear system
    # x = lstsq(signal_coef, general_coef, rcond=None)[0]
    # print(x)  

    #Haar transform
 


    # VMD clustering 
    # VMD_modes = 2
    # u = clustering.VMD_clustering(t, general, VMD_modes)

    # plt.figure()
    # plt.xlabel('t')
    # plt.ylabel('X (t)')
    # plt.title('VMD') 
    # for i in range(0, VMD_modes):
    #     plt.subplot(VMD_modes, 1, i+1)
    #     plt.plot(t, u[i, :])

    # plt.show()

