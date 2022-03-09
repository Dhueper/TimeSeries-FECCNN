#Test functions
from numpy import pi, arccos, arcsin, sin, cos, sqrt, linspace, zeros, array
from matplotlib import pyplot as plt 

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

if __name__ == "__main__":
    name = 'W_Lights'
    [t, X] = read('data/Sanse/20220301.plt', name)

    for j in range(0,10):
        for i in range(1,len(X)-1):
            X[i] = (X[i-1] + 2*X[i] + X[i+1])/4. 

     #Plots
    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title(name) 
    plt.show()
