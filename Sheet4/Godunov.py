import numpy as np
from math import pi
import matplotlib.pyplot as plt


def u0(x):
    return np.sin(4 * pi * x)


def con_flux(u):
    return 0.5 * u ** 2


def num_flux(u, v, con_flux):
    if u < v:
        theta = np.linspace(u, v)
        flux = np.min(con_flux(theta))
    else:
        theta = np.linspace(v, u)
        flux = np.max(con_flux(theta))
    return flux


def Godunov(nx, tf):
    a = -1
    b = +1
    x = np.linspace(a, b, nx + 1)
    dx = (b - a) / nx
    x = x[0:-1]  # periodic boundaries
    dt = 0.3 * dx
    t = 0
    u = u0(x)
    while t < tf:
        dt = np.min([dt, tf - dt])
        c = dt / dx
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] - c * (num_flux(un[i], un[i + 1], con_flux) - num_flux(un[i - 1], un[i], con_flux))

        u[0] = un[0] - c * (num_flux(un[0], un[1], con_flux) - num_flux(un[nx - 1], un[0], con_flux))
        u[nx - 1] = un[nx - 1] - c * (
                    num_flux(un[nx - 1], un[0], con_flux) - num_flux(un[nx - 2], un[nx - 1], con_flux))
        t = t + dt

    return x, u


nx = 100
tf = .5
x, u = Godunov(nx, tf)
plt.plot(x, u, '-o')
plt.show()
