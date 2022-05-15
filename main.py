import os
import sys
import random

from matplotlib import pyplot as plt
from numpy import zeros, mean, asfortranarray, linspace, amax, amin, ones, sum, array, transpose, sqrt, var, abs
from numpy.linalg import lstsq
from numpy import random
from scipy.interpolate import interp1d
from scipy.signal import spectrogram

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

def user_examples(N):
    """Pre-defined examples to introduce new users to time series feature extraction and classification:
    1) Spectral and statistical feature extraction.
    2) Bispectral transform.
    3) Spectrogram.
    4) Time series compression through the Haar transform.
    5) Haar compression error with the sampling rate.
    6) Haar series expansion.
    7) Haar Pattern Decomposition and Classification (HPDC).
    8) CNN classification (several methods).

    Intent(in): N(integer), example selected;

    Returns: function, example.
    """

    def plot(t,X):
        plt.figure()
        plt.plot(t,X)
        plt.xlabel('t')
        plt.ylabel('X(t)')
        plt.title('Original time series')

    def example1():
        """Time series spectral and statistical feature extraction.

        Intent(in): None

        Returns: None
        """

        print('Example 1: Spectral and statistical feature extraction.')

        [t, X] = test_function.sinusoidal_function()
        plot(t,X)

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

        plt.show()

    def example2():
        """Time series bispectral transform.

        Intent(in): None

        Returns: None
        """

        print('Example 2: Bispectral transform.')

        [t, X] = test_function.sinusoidal_function_f_mod()

        plot(t,X)

        bs = bispectrum.bispectral_transform(t, X)

        plt.figure()
        bs.plot_mag()
        plt.show()

    def example3():
        """Time series spectrogram.

        Intent(in): None

        Returns: None
        """

        print('Example 3: Spectrogram.')

        [t, X] = test_function.sinusoidal_function_f_mod()
        plot(t,X)

        fs = 1./(t[1] - t[0])
        f, t, Sxx = spectrogram(X, fs, nperseg=128)
        f_max = len(f)//3

        plt.pcolormesh(t, f[0:f_max], Sxx[0:f_max,:], shading='gouraud')
        plt.ylabel('$\it{f}$ [Hz]', rotation=0)
        plt.xlabel('$\it{t}$ [s]')
        plt.colorbar()
        plt.show()

    def example4():
        """Time series compression through the Haar transform.

        Intent(in): None

        Returns: None
        """

        print('Example 4: Compression through the Haar transform.')

        [t0, X] = test_function.sinusoidal_function()  
        plot(t0,X)

        t0 = t0 / amax(t0)

        #Reshape to fit a power of 2. 
        [t, Y] = rectangular_signal.reshape_2pow(t0, X) 

        #Haar transform 
        Y_h = haar.haar_1d ( len(Y), Y )

        plt.figure()
        plt.plot(t, Y_h)
        plt.xlabel('')
        plt.ylabel('$\it{H}$', rotation=0)
        plt.title('Haar Transform')

        #Compression
        comp_ratio = 1./2

        Y_h[int(comp_ratio*len(Y_h)):-1] = 0.0
        Y_inv =   haar.haar_1d_inverse (len(Y_h), Y_h)

        plt.figure()
        plt.plot(t, Y_inv)
        plt.xlabel('$\it{t}$ [s]')
        plt.ylabel('$\it{X(t)}$', rotation=0)
        plt.title('Reconstructed signal with' + str(int(1./comp_ratio)) + ':1 compression')
        plt.show()

    def example5():
        """Time series Haar compression error as a function of the sampling rate.

        Intent(in): None

        Returns: None
        """

        print('Example 5: Haar compression error.')

        s_rate = array([2**i for i in range(10, 19)]) # Sampling rate 
        rmse = [] 
        error = [] 
        max_comp = 4 #Maximum compression ratio will be 2**max_comp 
        plt.figure()

        for r in s_rate:
            [t0, X] = test_function.sinusoidal_function_rate(r)  
            t0 = t0 / amax(t0)

            #Reshape to fit a power of 2. 
            [t, Y] = rectangular_signal.reshape_2pow(t0, X) 

            Y_h = haar.haar_1d ( len(Y), Y ) # Haar transform 

            #Compression
            rmse = [] 
            for j in range(1, max_comp+1):
                comp_ratio = 1./2**j
                Y_h[int(comp_ratio*len(Y_h)):-1] = 0.0
                Y_inv =   haar.haar_1d_inverse (len(Y_h), Y_h)
                rmse.append(sqrt(sum((Y-Y_inv)**2.)/len(Y)))
            error.append(rmse)

        error = array(error).T
        legend = [] 
        for j in range(0,max_comp):
            plt.semilogy(s_rate, error[j,:])
            legend.append(str(int(2**(j+1))) + ':1')
        plt.xlabel('$\it{sampling \,  rate}$ [Hz]')
        plt.ylabel('$\it{RMSE}$', rotation=0)
        plt.title('Compression error for different compression ratios')
        plt.legend(legend)
        plt.show()
        

    def example6():
        """Time series Haar series expansion.

        Intent(in): None

        Returns: None
        """

        print('Example 6: Haar series expansion.')

        name = 'W_Gas_boiler'
        [t0, X] = test_function.read('data/Sanse/20220301.plt', name) 
        plot(t0,X)

        t0 = t0 / amax(t0)
        Z = zeros(len(X))
        Z[:] = X[:]  

        #Mean value filter from: https://github.com/Dhueper/TimeSeries-AnomalyDetection
        for _ in range(0,50):
            Z = fortran_ts.time_series.mvf(asfortranarray(Z), 0)

        #Reshape to fit a power of 2. 
        [t, Y] = rectangular_signal.reshape_2pow(t0, Z) 

        #Haar series expansion
        order =  6
        c_haar = rectangular_signal.haar_coef(t, Y, order)
        N = len(t)
        c = zeros(N)
        Y_haar = ones(N)
        Y_haar = Y_haar * mean(Y) 

        for m in range(0, order):
            for n in range(0, 2**m):
                for i in range(0, N):
                    c[i] = rectangular_signal.phi(t[i], m, n) * c_haar[m][n]  
                Y_haar  = Y_haar + c 

        plt.figure()
        plt.plot(t0, X, 'g')
        plt.plot(t, Y, 'c')
        plt.plot(t, Y_haar, 'b')
        plt.xlabel('t [h]')
        plt.ylabel('P [W]')
        plt.title('Power consumption')
        plt.legend(['Original time series', 'Filtered time series', 'Haar expansion'])

        N_coef = sum(array([2**i for i in range(0,order)])) + 1
        print('Haar signal:', N_coef, 'coefficients')
        print('Haar coefficients:')
        print(c_haar)

        plt.show()

    def example_invalid():
        print('Invalid case selected. Select an example from 1 to 8.')

    #Switch case dictionary 
    switcher = {1: example1, 2:example2, 3:example3, 4:example4, 5:example5, 6:example6, 7:example7, 8:example8, 9:example9, 10:example10}
    #Get the function from switcher dictionary  
    example = switcher.get(N, example_invalid)

    return example()

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


def power_estimation():
    name_list = {'W_Air_cond':4, 'W_Gas_boiler':4} 
    signal_coef = [] 
    signal_general = [] 
    signal_ratio = [] 
    mean_ratio = []
    var_ratio = []
    rmse_ratio = []
    order = 4
    ct = 0
    use_env = False
    for name in name_list.keys():
        plt.figure()
        plt.title(name)
        plt.xlabel('t [h]')
        plt.ylabel('P [W]')

        for i in range(0,2):
            [t0, X] = test_function.read('data/Sanse/2022030'+str(1 + 7*i)+'.plt', name) 
            if i == 0:
                ## Use envelope as pattern
                if use_env:
                    env = zeros((7, len(X)))
                    env[0,:]  = array(X)[:] 
                    X2 = zeros(len(X))
                    for j in range(1, 7):
                        [t1, X1] = test_function.read('data/Sanse/2022030'+str(1 + 7*i + j)+'.plt', name)
                        X2[0:min(len(X1), len(X2))]  = array(X1)[0:min(len(X1), len(X2))] 
                        env[j,:] = X2[:]  
                    max_env = amax(env, axis=0)
                    min_env = amin(env, axis=0)
                    mean_env = mean(env, axis=0)
                    X[:] = mean_env[:]  

                # plt.plot(t0, max_env)
                # plt.plot(t0, min_env)
                # plt.plot(t0, mean_env)

            elif i == 1:
                # r = random.randint(0, int(30*len(X)/(24*60))) #Shift phase up to 30 min
                # p = 1 + (random.randint(0, 10) - 5)/10.
                r = 0
                p = 1

                X1 = zeros(len(X))
                X1[:] = X[:]
                X[r:len(X)] = X1[0:len(X)-r]   
                X[0:r] = X1[len(X)-r:len(X)]   
                X = p * X
                 

            t0 = t0 / amax(t0)
            Z = zeros(len(X))
            Z[:] = X[:]  
            #Mean value filter 
            for _ in range(0,50):
                Z = fortran_ts.time_series.mvf(asfortranarray(Z), 0)

            #Reshape to fit a power of 2. 
            [t, Y] = rectangular_signal.reshape_2pow(t0, Z)

            if i == 0: # Reference signal of class 'name'  

                #Haar series expansion
                c_haar = rectangular_signal.haar_coef(t, Y, order)

                signal_coef.append([])
                signal_coef[ct].append(mean(Y)) 

                signal_ratio.append(Y)
                mean_ratio.append(mean(Y))
                var_ratio.append(var(Y))
                rmse_ratio.append(Y)

                for m in range(0, order):
                    for n in range(0, 2**m):
                        # for i in range(0, N):
                        #     c[i] = rectangular_signal.phi(t[i], m, n) * c_haar[m][n]  
                        signal_coef[ct].append(c_haar[m][n]) 


            else:
                signal_general.append(Y)
                signal_ratio[ct] = 1. / ( sum(signal_ratio[ct] / Y) / len(Y) )
                mean_ratio[ct] = 1. / ( mean_ratio[ct] / mean(Y) )
                var_ratio[ct] = 1. / ( var_ratio[ct] / var(Y) )
                # rmse_ratio[ct] =   sqrt(sum((signal_ratio[ct] - Y)**2.) / len(Y)) 
                rmse_ratio[ct] = sqrt(sum(Y**2.) / len(Y)) / sqrt(sum(rmse_ratio[ct]**2.) / len(Y)) 

            plt.plot(t, Y)
            
        ct += 1
        plt.legend(['mean_env', 'signal'])
        plt.show()
                
    #Reference signals matrix 
    signal_coef = transpose(array(signal_coef))
    print(type(signal_coef), signal_coef.shape)

    #General power signal 
    general = array(signal_general[0][:])
    for i in range(1, len(signal_general)):
        general = general + array(signal_general[i][:])

    c_haar = rectangular_signal.haar_coef(t, general, order)
    general_coef = [mean(general)]
    for m in range(0, order):
        for n in range(0, 2**m):
              general_coef.append(c_haar[m][n])
    general_coef = transpose(array(general_coef))
    print(type(general_coef), general_coef.shape)

    #Solve linear system
    x = lstsq(signal_coef, general_coef, rcond=None)[0]
    print('System solution:', x) 
    print('Mean ratio:', mean_ratio)
    print('Signal ratio:', signal_ratio)
    print('sigma ratio:', sqrt(var_ratio))
    print('RMSE ratio:', rmse_ratio)
    return abs(x - array(mean_ratio)), abs(x - array(signal_ratio))



if __name__ == "__main__":

    run = True
    while run:
        print(""" Select an introductory pre-defined example:\n 
        0) Exit\n 
        1) Spectral and statistical feature extraction.\n 
        2) Bispectral transform.\n 
        3) Spectrogram.\n 
        4) Time series compression through the Haar transform.\n 
        5) Haar compression error with the sampling rate.\n 
        6) Haar series expansion.\n 
        7) Haar Pattern Decomposition and Classification (HPDC).\n 
        8) CNN classification (several methods).\n 
        """)

        option = input("Select an example from 0 to 8: ")
        if option == '0':
            run = False
        else:
            user_examples(int(option))
        run = True


    #Run examples 

    # VMD_example()

    # EWT_example()

    # EMD_example()

    # power_estimation()




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
    #     plt.show()

    # signal_coef = transpose(array(signal_coef))
    # print(type(signal_coef), signal_coef.shape)

    # general = 0.4*signal[0] + 0.6*signal[1] + signal[2] 
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

