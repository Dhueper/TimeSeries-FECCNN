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

    return[t,y] 

if __name__ == "__main__":
    [t, X] = triangle_function()

     #Plots
    plt.figure()
    plt.plot(t,X)
    plt.xlabel('t')
    plt.ylabel('X (t)')
    plt.title('Test function') 
    plt.show()
