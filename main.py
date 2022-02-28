import os
import sys

from matplotlib import pyplot as plt

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import bispectrum

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


