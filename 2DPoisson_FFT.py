import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import pi


P = pi #domain size in x
Q = pi/2 #domain size in y


def fourier(M, N, f):
    x = np.linspace(0,pi,M + 1)
    y = np.linspace(0,pi/2,N + 1)
    x = x[0:-1]
    y = y[0:-1] #remove last points
    
    Lx = len(x)
    Ly = len(y)
    ind_k = (2*pi/P) * np.array([i for i in range(0, Lx//2)] + [i for i in range(-Lx//2,0)])
    ind_m = (2*pi/Q) * np.array([i for i in range(0, Ly//2)] + [i for i in range(-Ly//2,0)])
    [Kx, Ky] = np.meshgrid(ind_k, ind_m)
    denom = np.square(Kx) + np.square(Ky)
    denom[0,0] = 1 #avoid wavenumber (0,0)
    [X, Y] = np.meshgrid(x,y)
    fhat = fft2(f(X,Y))
    uhat = fhat/denom
    uhat[0,0] = 0 #zero mean solution
    u = np.real(ifft2(uhat))
    
    return X, Y, u


def f(x,y):
    return np.sin(x)

M = 64
N = 32
X,Y, u = fourier(M, N, f)

# visualization
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, u, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Poisson equation via spectral method');
plt.xlabel('x')
plt.ylabel('y')

plt.matshow(u)
plt.grid()
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')