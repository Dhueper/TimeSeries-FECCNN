import os
import sys

from matplotlib import pyplot as plt

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum
import clustering
import feature_extraction

#%%Bispectrum 
def bispectrum_example():
    #Square function transformation 
    [t, X] = test_function.square_function()

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Square function') 

    bs = bispectrum.bispectral_transform(t, X)

    plt.figure()
    bs.plot_mag()
    # plt.show()

    #Triangle function transformation 

    [t, Y] = test_function.triangle_function()

    plt.figure()
    plt.plot(t,Y)
    plt.xlabel('t')
    plt.ylabel('Y (t)')
    plt.title('Triangle function') 

    bs = bispectrum.bispectral_transform(t, Y)

    plt.figure()
    bs.plot_mag()
    # plt.show()

    #Sum of square + triangle functions
    #  
    plt.figure()
    plt.plot(t,X+Y)
    plt.xlabel('t')
    plt.ylabel('X+Y (t)')
    plt.title('Square + Triangle function') 

    bs = bispectrum.bispectral_transform(t, X+Y)

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
    [t, X] = test_function.square_function()
    [t, Y] = test_function.triangle_function()

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
    [t, X] = test_function.square_function()
    [t, Y] = test_function.triangle_function()

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
    [t, X] = test_function.square_function()
    [t, Y] = test_function.triangle_function()

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

    SPW = feature_extraction.SPow(t,X)
    print("Spectral Power: SPW=", SPW)

    SE = feature_extraction.SEnt(t,X)
    print("Spectral Entropy: SE=", SE)

    SP, fP = feature_extraction.SPeak(t,X)
    print("Spectral Peak: SP=", SP)
    print("Peak frequency: fP=", fP)

    SC = feature_extraction.SCen(t,X)
    print("Spectral Centroid: SC=", SC)

    AM, FM = feature_extraction.BW(t,X)
    print("AM bandwidth: AM=", AM)
    print("FM bandwidth: FM=", FM)

    V, HM, HC = feature_extraction.Hjorth(t,X)
    print("Variance: Var=", V)
    print("Hjorth Mobility: HM=", HM)
    print("Hjorth Complexity: HC=", HC)

    SK = feature_extraction.Skew(t,X)
    print("Skewness: SK=", SK)

    KT = feature_extraction.Kurt(t,X)
    print("Kurtosis: KT=", KT)


if __name__ == "__main__":
    #Run examples 
    # bispectrum_example()

    # VMD_example()

    # EWT_example()

    # EMD_example()

    feature_extraction_example()
