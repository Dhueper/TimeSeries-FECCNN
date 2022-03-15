import sys
import os

from matplotlib import pyplot as plt
from numpy import zeros, mean

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum

[t, X] = test_function.read('data/Sanse/20220301.plt', 'W_Lights')

plt.figure()
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X (t)')
plt.title('Power [W]') 

bs = bispectrum.bispectral_transform(t, X)

print(bs.bispec_mag.shape)

plt.figure()
bs.plot_mag()
plt.show()
