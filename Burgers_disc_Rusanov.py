import numpy as np
import matplotlib.pyplot as plt


#Solve inviscid Burger's equation by Rusanov flux with different discontinuous initial data


def f(u):
    return 0.5 * u**2  #Burger flux


def df(u):
    return u


def flux(u,v,f,df):
    s = np.max([np.abs(df(u)), np.max(df(v))]) 
    return 0.5*(f(u) + f(v)) - 0.5*s*(v - u) #Rusanov flux


def u0(x,ul,ur):
    return np.piecewise(x, [x < 0, x >= 0], [lambda x: ul, lambda x: ur])


T = .5
N = 80
x = np.linspace(-1,1,N + 1)
dx = 2/N
dt = 0.7*dx #CFL
c = dt/dx
ts = np.int(T/dt)


### SHOCK WAVE with periodicity### 
x = x[0:-1] #periodic boundary conds: domain is [x_0,...,x_{N-1}], identify x_0=x_N
u = u0(x,1,0) #Shock
for n in range(ts+1):
    un = u.copy()
    for i in range(1,N-1):
        u[i] = un[i] - c*(flux(un[i],un[i+1],f,df) - flux(un[i-1],un[i],f,df))

     #periodic boundaries: solve at internal knots: x_1,...x_{N-2}
     #and update by u[-1] = u[N-1], u[N] = u[0]
    u[0] = un[0] - c*(flux(un[0],un[1],f,df) - flux(un[N-1],un[0],f,df))
    u[N-1] = un[N-1] - c*(flux(un[N-1],un[0],f,df) - flux(un[N-2],un[N-1],f,df))
           
plt.figure()
plt.plot(x,u,'r-o',markerfacecolor='None')
plt.xlabel('x')
plt.title('Sol at T={}s '.format(np.round(n*dt,3)) + '(shock wave)')



### SHOCK WAVE### 
x = np.linspace(-1,1,N + 1)
u = u0(x,1,0) #Shock
for n in range(ts+1):
    un = u.copy()
    for i in range(1,N):
        u[i] = un[i] - c*(flux(un[i],un[i+1],f,df) - flux(un[i-1],un[i],f,df))

    u[0] = 1
    u[N] = 0

plt.figure()
plt.plot(x,u,'b-o',markerfacecolor='None')
plt.xlabel('x')
plt.title('Sol at T={}s '.format(np.round(n*dt,3)) + '(shock wave)')



### RAREFACTION WAVE ###
x = np.linspace(-1,1,N + 1)
u = u0(x,-1,1) #Rarefaction wave
for n in range(ts+1):
    un = u.copy()
    for i in range(1,N):
        u[i] = un[i] - c*(flux(un[i],un[i+1],f,df) - flux(un[i-1],un[i],f,df))
    u[0] = -1 #Dirichlet
    u[N] = +1
        
    
plt.figure()
plt.plot(x,u,'g-o',markerfacecolor='None')
plt.xlabel('x')
plt.title('Sol at T={}s '.format(np.round(n*dt,3)) + '(rarefaction wave)')