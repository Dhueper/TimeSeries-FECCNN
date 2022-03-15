import os
import sys

from matplotlib import pyplot as plt
from numpy import zeros, mean

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

    Features = feature_extraction.Features(t, X)

    SPW = Features.SPow()
    print("Spectral Power: SPW=", SPW)

    SE = Features.SEnt()
    print("Spectral Entropy: SE=", SE)

    SP, fP = Features.SPeak()
    print("Spectral Peak: SP=", SP)
    print("Peak frequency: fP=", fP)

    SC = Features.SCen()
    print("Spectral Centroid: SC=", SC)

    AM, FM = Features.BW()
    print("AM bandwidth: AM=", AM)
    print("FM bandwidth: FM=", FM)

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

    # feature_extraction_example()


    #Try CNN network
     


 

    # [t, Xc] = test_function.read('data/Sanse/20220301.plt', 'W_Lights')
    # [t, Xg] = test_function.read('data/Sanse/20220301.plt', 'W_Gas_boiler')

    # for j in range(0,10):
    #     for i in range(1,len(t)-1):
    #         Xc[i] = (Xc[i-1] + 2*Xc[i] + Xc[i+1])/4. 
    #         Xg[i] = (Xg[i-1] + 2*Xg[i] + Xg[i+1])/4. 

    # Y = Xc + Xg 

    # VMD_modes = 2
    # u = clustering.VMD_clustering(t, Y, VMD_modes)
    # if len(u[0,:]) < len(t):
    #     t1 = zeros(len(u[0,:]))
    #     t1[:] = t[:-1] 
    # else:
    #     t1 = zeros(len(t))
    #     t1[:] = t[:]   

    # bs = bispectrum.bispectral_transform(t, Xc)
    # plt.figure()
    # bs.plot_mag()
    # plt.title('Xc')
    # bs = bispectrum.bispectral_transform(t, Xg)
    # plt.figure()
    # bs.plot_mag()
    # plt.title('Xg')
    # bs = bispectrum.bispectral_transform(t1, u[0,:])
    # plt.figure()
    # bs.plot_mag()
    # plt.title('Mode 1')
    # bs = bispectrum.bispectral_transform(t1, u[1,:])
    # plt.figure()
    # bs.plot_mag()
    # plt.title('Mode 2')

    # plt.figure()
    # plt.title('Power [W]')
    # plt.subplot(2,1,1)
    # plt.plot(t,Xc,'b')
    # plt.xlabel('t')
    # plt.ylabel('W_computer')

    # plt.subplot(2,1,2)
    # plt.plot(t,Xg,'r')
    # plt.xlabel('t')
    # plt.ylabel('W_Gas_boiler')


    # plt.figure()
    # plt.plot(t,Y)
    # plt.xlabel('t')
    # plt.ylabel('W_both')
    # plt.title('Power [W]') 

    

    # plt.figure()
    # plt.xlabel('t')
    # plt.ylabel('X (t)')
    # plt.title('VMD') 
    # for i in range(0, VMD_modes):
    #     plt.subplot(VMD_modes, 1, i+1)
    #     plt.plot(t1, u[i, :])

    # plt.show()

