import sys
import os
import time

from matplotlib import pyplot as plt
from numpy import zeros, mean, linspace, array, linalg, save, asfortranarray
from scipy.interpolate import interp1d

from keras.utils import to_categorical

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum
import fortran_ts

def process_raw_dataset():

    t0 = time.time()

    [X, Y] = test_function.read_dataset('ElectricDevices/ElectricDevices_TEST.txt') 

    t = linspace(0, 24*60, len(X[0,:]))

    X_test = [] 
    Y_test = [] 

    for i in range(0, len(X[:,0])):
        if Y[i] < 6 and Y[i]>1: 
            bs = bispectrum.bispectral_transform(t, X[i,:])
            bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)
            X_test.append(bs.bispec_mag)
            Y_test.append(Y[i]-1)

    X_test = array(X_test)
    X_test = X_test.reshape(len(X_test),21,21,1)

    Y_test = array(Y_test)
    Y_test = Y_test -1.0
    Y_test = to_categorical(Y_test)

    print(X_test.shape)

    print(Y_test.shape)

    tf = time.time()

    print('t=', tf-t0)

    # save('ElectricDevices/X_test.npy', X_test)
    # save('ElectricDevices/Y_test.npy', Y_test)

def process_raw_data():

    t0 = time.time()

    tags = ['W_Air_cond','W_Computers','W_Audio_TV','W_Lights','W_Kitchen','W_Washing_m','W_Dish_w','W_Gas_boiler','W_Oven_vitro'] 

    for tag in tags:
        X_eval = []
        for i in range(1,9):
            [t, X] = test_function.read('data/Sanse/2022030'+str(i)+'.plt', tag) 

            [t95, X95] = process_signal(t, X) 

            bs = bispectrum.bispectral_transform(t95, X95)
            bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)
            X_eval.append(bs.bispec_mag)

        X_eval = array(X_eval)
        X_eval = X_eval.reshape(i,21,21,1)

        # save('ElectricDevices/X_eval_'+tag+'.npy', X_eval)


        print(X_eval.shape)

    tf = time.time()

    print('t=', tf-t0)

def process_autoencoder_dataset():
    t0 = time.time()
    tag = 2

    [X, Y] = test_function.read_dataset('ElectricDevices/ElectricDevices_TEST.txt') 

    t = linspace(0, 24*60, len(X[0,:]))

    X_test = [] 
    Y_test = [] 
    ct = 0
    for i in range(0, len(X[:,0])):
        if Y[i] == tag:
            X1 = X[i,:]
            bs1 = bispectrum.bispectral_transform(t, X1)
            bs1.bispec_mag = bs1.bispec_mag/linalg.norm(bs1.bispec_mag)
            for j in range(0, len(X[:,0])):
                if Y[j] == tag+1:
                    X2 = X[j,:]
                    X_total = X1 + X2
                    bs = bispectrum.bispectral_transform(t, X_total)
                    bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)
                    X_test.append(bs.bispec_mag)
                    Y_test.append(bs1.bispec_mag)
                    ct += 1
                if ct >= 10000:
                    break
        if ct >= 10000:
                    break


    X_test = array(X_test)
    Y_test = array(Y_test)
    X_test = X_test.reshape(len(X_test), 21, 21, 1)
    Y_test = Y_test.reshape(len(Y_test), 21, 21, 1)

    print(X_test.shape)
    print(Y_test.shape)

    save('ElectricDevices/X_test_AE_'+str(tag)+'.npy', X_test)
    save('ElectricDevices/Y_test_AE_'+str(tag)+'.npy', Y_test)

    tf = time.time()

    print('t=', tf-t0)

def process_autoencoder_data():
    t0 = time.time()

    tags = ['W_Air_cond','W_Computers','W_Audio_TV','W_Lights','W_Kitchen','W_Washing_m','W_Dish_w','W_Gas_boiler','W_Oven_vitro'] 
    tags1 = ['W_Air_cond'] 
    key_tag = 'W_Air_cond'
    X_test = []
    Y_test = [] 
    for i in range(1,9):
        for tag1 in tags1:
            [t, X] = test_function.read('data/Sanse/2022030'+str(i)+'.plt', tag1) 

            [t1_95, X1_95] = process_signal(t, X) 

            bs1 = bispectrum.bispectral_transform(t1_95, X1_95)
            bs1.bispec_mag = bs1.bispec_mag/linalg.norm(bs1.bispec_mag)

            for tag2 in tags:
                if tag2 != key_tag and tag2 != tag1:
                    [t, X] = test_function.read('data/Sanse/2022030'+str(i)+'.plt', tag2) 

                    [t2_95, X2_95] = process_signal(t, X) 

                    X95 = X1_95 + X2_95

                    bs = bispectrum.bispectral_transform(t1_95, X95)
                    bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)

                    X_test.append(bs.bispec_mag)
                    if tag1 == key_tag:
                        Y_test.append(bs1.bispec_mag)
                    else:
                        Y_test.append(zeros((21,21)))

    X_test = array(X_test)
    Y_test = array(Y_test)
    X_test = X_test.reshape(len(X_test), 21, 21, 1)
    Y_test = Y_test.reshape(len(Y_test), 21, 21, 1)

    print(X_test.shape)
    print(Y_test.shape)

    # save('ElectricDevices/X_train_AE_'+key_tag+'.npy', X_test)
    # save('ElectricDevices/Y_train_AE_'+key_tag+'.npy', Y_test)

    tf = time.time()

    print('t=', tf-t0)

def process_signal(t, X):
    #Noise Mean Value Filter
    for _ in range(0,10):
        X = fortran_ts.time_series.mvf(asfortranarray(X), 2)
        X[0] = 2*X[1] - X[2] 
        X[len(X)-1] = 2*X[len(X)-2] - X[len(X)-3] 

    #Interpolation 
    f = interp1d(t*60, X, fill_value='extrapolate')

    t95 = linspace(0, 24*60, 95)
    X95 = f(t95)

    return [t95, X95] 

if __name__ == "__main__":
    # process_raw_dataset()
    # process_raw_data()
    # process_autoencoder_data()
    process_autoencoder_dataset()
