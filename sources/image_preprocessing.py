import sys
import os
import time

from matplotlib import pyplot as plt
from numpy import zeros, mean, linspace, array, linalg, save
from keras.utils import to_categorical

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum

def process_raw_dataset():

    t0 = time.time()

    [X, Y] = test_function.read_dataset('ElectricDevices/ElectricDevices_TEST.txt') 

    t = linspace(0, 24*60, len(X[0,:]))

    X_test = [] 

    for i in range(0, len(X[:,0])):
        bs = bispectrum.bispectral_transform(t, X[i,:])
        bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)
        X_test.append(bs.bispec_mag)

    X_test = array(X_test)
    X_test = X_test.reshape(len(X[:,0]),21,21,1)

    Y = Y -1.0
    Y_test = to_categorical(Y)

    print(X_test.shape)

    print(Y_test.shape)

    tf = time.time()

    print('t=', tf-t0)

    # save('ElectricDevices/X_test.npy', X_test)
    # save('ElectricDevices/Y_test.npy', Y_test)

def process_raw_data():

    t0 = time.time()

    tags = ['W_Air_cond','W_Computers','W_Audio_TV','W_Lights','W_Kitchen','W_Washing_m','W_Dish_w','W_Gas_boiler','W_Oven_vitro'] 

    X_eval = [] 
    for tag in tags:
        [t, X] = test_function.read('data/Sanse/20220301.plt', tag) 

        bs = bispectrum.bispectral_transform(t, X)
        bs.bispec_mag = bs.bispec_mag/linalg.norm(bs.bispec_mag)
        X_eval.append(bs.bispec_mag)

    X_eval = array(X_eval)
    X_eval = X_eval.reshape(len(tags),21,21,1)


    print(X_eval.shape)

    tf = time.time()

    print('t=', tf-t0)

    # save('ElectricDevices/X_eval.npy', X_eval)

if __name__ == "__main__":
    # process_raw_dataset()
    # process_raw_data()
