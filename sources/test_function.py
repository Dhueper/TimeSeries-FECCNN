#Test functions
import sys
import os

from numpy import pi, arccos, arcsin, sin, cos, sqrt, linspace, zeros, array, float32, asfortranarray, concatenate
from numpy.random import normal
from matplotlib import pyplot as plt 
from scipy.interpolate import interp1d

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import fortran_ts

def square_function():
    t = linspace(0,100,100)
    y = zeros(len(t))
    for i in range(0,int(len(t)/2)):
        if i%2 == 1:
            y[2*i] = 1
            y[2*i+1] = 1
        else:
            y[2*i] = -1 
            y[2*i+1] = -1 

    return [t,y] 

def square_function2():
    t = linspace(0,100,1000)
    y = zeros(len(t))
    for i in range(0,int(len(t)/10)):
        if i%2 == 1:
            for j in range(0,10):
                y[10*i+j] = 2

        else:
            for j in range(0,10):
                y[10*i+j] = -2

    return [t,y]

def square_function3():
    t = linspace(0,100,1000)
    y = zeros(len(t))
    for i in range(0,int(len(t)/5)):
        if i%2 == 1:
            for j in range(0,5):
                y[5*i+j] = 1

        else:
            for j in range(0,10):
                y[5*i+j] = -1

    return [t,y]

def triangle_function():
    t = linspace(0,100,100)
    y = zeros(len(t))
    for i in range(0,int(len(t)/2)):
        y[2*i] = 1
        y[2*i+1] = -1  

    return [t, y] 

def sinusoidal_function():
    t = linspace(0,1,1000)
    y = zeros(len(t))

    y = 4*sin(2*pi*10 * t) + 2*sin(2*pi*30 * t) + 1*sin(2*pi*50 * t)

    return [t,y] 

def sinusoidal_function_rate(n):
    t = linspace(0,1,n)
    y = zeros(len(t))

    y = 4*sin(2*pi*10 * t) + 2*sin(2*pi*30 * t) + 1*sin(2*pi*50 * t)

    return [t,y] 

def sinusoidal_function_f_mod():
    t = linspace(0,1,1000)
    y = zeros(len(t))

    y = sin(2*pi*10*(t+1) * t) + sin(2*pi*50*(t+1) * t)

    return [t,y] 

def read(filename, name='W_General'): 
  file = open(filename,'r')
  N = len(file.readlines())
#   N = 1000
  file.seek(0)
  time = zeros(N-1)
  x = zeros(N-1)
  vars = file.readline().split('=')[1].split(',')
  index = vars.index(name)

  for i in range(0,N-1):
    line = array(file.readline().split(','))
    time[i] = float(line[0])
    x[i] = float(line[index])

  file.close()
  return time, x

def read_dataset(filename):
    file = open(filename,'r')
    N = len(file.readlines())
    # N = 100
    file.seek(0)
    Y = []
    X = []  
    for i in range(0,N):
        line = file.readline().split('\n')[0].split('  ')
        Y.append(float(line[1].split(' ')[-1]))
        X.append(array(line[2:-1], dtype=float32))
    X = array(X)
    Y = array(Y)

    file.close()
    return [X, Y] 

def read_mars(filename, name='NPWD2372'): 
  file = open(filename,'r')
#   N = len(file.readlines())
  N = 1000
  file.seek(0)
  time = zeros(N-1)
  x = zeros(N-1)
  vars = file.readline().split(',')
  index = vars.index(name)

  for i in range(0,N-1):
    line = array(file.readline().split(','))
    time[i] = float(line[0])
    x[i] = float(line[index])

  file.close()
  time = (time - time[0])/1000 
  return time, x

def spectral_entropy():
    t = linspace(0,1,1000)
    y = zeros(len(t))

    y = normal(0, 0.1, len(y)) + sin(2*pi*10 * t)
    x = normal(0, 1, len(y)) 

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, x, 'r')
    plt.ylabel('$\it{N (t)}$', rotation=0, fontsize=18)
    plt.legend(['SEnt=0.98'], fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.subplot(2,1,2)
    plt.plot(t, y, 'b')
    plt.xlabel('$\it{t}$ [s]', fontsize=18)
    plt.ylabel('$\it{X (t)}$', rotation=0, fontsize=18)
    plt.legend(['SEnt=0.22'], fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    return [t,x] 

def bandwidth_am():
    t = linspace(0,1,1000)
    y = 4*sin(2*pi*10 * t) + 2*sin(2*pi*30 * t) + 1*sin(2*pi*60 * t)
    x = 4*sin(2*pi*10 * t) + 2*sin(2*pi*40 * t) + 1*sin(2*pi*80 * t)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, x, 'r')
    plt.ylabel('$\it{X_1 (t)}$', rotation=0, fontsize=18)
    plt.legend(['B_AM=160 Hz, B_FM=495 Hz'], fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.subplot(2,1,2)
    plt.plot(t, y, 'b')
    plt.xlabel('$\it{t}$ [s]', fontsize=18)
    plt.ylabel('$\it{X_2 (t)}$', rotation=0, fontsize=18)
    plt.legend(['B_AM=127 Hz, B_FM=354 Hz'], fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    return [t,y]

if __name__ == "__main__":
    # [t, X] = read_mars('data/mars-express-power-3years/train_set/power--2008-08-22_2010-07-10.csv', name='NPWD2451')
    name = 'W_Gas_boiler'
    [t, X] = read('data/Sanse/20220301.plt', name)

    for _ in range(0,50):
        X = fortran_ts.time_series.mvf(asfortranarray(X), 0)

    plt.figure()
    plt.plot(t,X)
    plt.xlabel('$\it{t}$ [h]', fontsize=18)
    plt.ylabel('$\it{P}$ [W]', fontsize=18, rotation=0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['Pattern 2'], loc='upper left')
    plt.show()

    # bandwidth_am()

    # [t, X] = square_function3() 
    # [t, Y] = square_function2() 
    # Z = X + Y

    # plt.figure()
    # plt.plot(t, Z)
    # plt.show() 

    # name = 'W_Lights'
    # [t, X] = read('data/Sanse/20220301.plt', name)

    # #Noise Mean Value Filter
    # for _ in range(0,10):
    #     X = fortran_ts.time_series.mvf(asfortranarray(X), 2)
    #     X[0] = 2*X[1] - X[2] 
    #     X[len(X)-1] = 2*X[len(X)-2] - X[len(X)-3] 

    # f = interp1d(t*60, X, fill_value='extrapolate')

    # t95 = linspace(0, 24*60, 95)
    # X95 = f(t95)

    #  #Plots
    # plt.figure()
    # plt.plot(t95,X95)
    # plt.xlabel('t')
    # plt.ylabel('X (t)')
    # plt.title(name) 
    # plt.show()
    # print(len(X95))

    # [X, Y] = read_dataset('ElectricDevices/ElectricDevices_TEST.txt') 
    # plt.figure()
    # plt.plot(X[0,:])
    # plt.show()
    # print(len(X[0,:] ))



    # name = 'W_Air_cond' 
    # for i in range(0, 7):
    #     [t0, X] = read('data/Sanse/2022030'+str(1 + i)+'.plt', name)
    #     if i == 0:
    #         Y = zeros(len(X))
    #         Y[:] = X[:]
    #         t = zeros(len(t0))
    #         t[:] = t0[:]
    #     else:
    #         Y = concatenate((Y, X), axis=None) 
    #         t = concatenate((t, t0 + t[-1]), axis=None) 

    # #Noise Mean Value Filter
    # for _ in range(0,100):
    #     Y = fortran_ts.time_series.mvf(asfortranarray(Y), 1)
    #     Y[0] = 2*Y[1] - Y[2] 
    #     Y[len(Y)-1] = 2*Y[len(Y)-2] - Y[len(Y)-3] 

    # plt.figure()
    # plt.plot(t, Y)
    # plt.xlabel('$\it{t}$ [h]', fontsize=18)
    # plt.ylabel('$\it{P}$ [W]', fontsize=18, rotation=0)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()



