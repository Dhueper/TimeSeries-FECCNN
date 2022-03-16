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

Y_test = to_categorical(Y)

print(X_test.shape)

print(Y_test.shape)

tf = time.time()

print('t=', tf-t0)

# save('ElectricDevices/X_test.npy', X_test)
# save('ElectricDevices/Y_test.npy', Y_test)

