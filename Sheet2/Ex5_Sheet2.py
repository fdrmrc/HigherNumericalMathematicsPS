import numpy as np
from math import pi
import matplotlib.pyplot as plt


#Convergence plot
tf=0.5
def u0(x):
    return np.sin(2*pi*x)
    #return np.exp(-50*np.power((x-1/2),2))

def convergence_plot(nsteps,method):
    #nsteps -> number of steps in x
    #method -> function of the method
    err=[]
    for nx in nsteps:
        x=np.linspace(0,1,nx+1)
        dx=x[2]-x[1]
        dt=0.5*dx #prescribed CFL
        u=method(nx,dt,tf)
        err.append(np.linalg.norm(u-u0(x-tf),np.Inf))
    
    plt.figure(figsize=(7,7))
    plt.loglog(nsteps,err,'-o',label='Error')
    plt.loglog(nsteps,err[-1]*(1/(nsteps/nsteps[-1])),'-',label='Order 1')
    plt.loglog(nsteps,err[-1]*1/((nsteps/nsteps[-1])**2),'-',label='Order 2')
    plt.legend()
    plt.title('Error plot for ' + str(method))
    plt.show()
    
   
def upwind(nx,dt,tf):
    nt=int(np.ceil(tf/dt))
    x=np.linspace(0,1,nx+1)
    dx=x[2]-x[1]
    t=0
    c=dt/dx
    if(dt>dx):
        print('cfl non fulfilled, c=',c)
    #else:
       #print('cfl=',c)
    u=u0(x)
    while(t< tf):#for n in range(0,nt):
        dt=np.min([tf-t,dt])
        c=dt/dx    
        un=u.copy()
        for i in range(1,nx+1): #1,...,nx: nx points
            u[i]=un[i]-c*(un[i]-un[i-1])
        u[0]=u[nx]
        
        if(t+dt>tf):
            print('dt resized since excedeed final time ') 
            dt=tf-t 
        t=t+dt
        #print(t,dt)
    return u

nx=83
dt=0.01
tf=0.2
x=np.linspace(0,1,nx+1)
u=upwind(nx,dt,tf)
plt.plot(x,u,'-o',markerfacecolor='none')
plt.plot(x,u0(x-tf),'-')
plt.show()



def LW(nx,dt,tf):
    nt=int(np.ceil(tf/dt))
    x=np.linspace(0,1,nx+1)
    dx=x[2]-x[1]
    c=dt/dx    
    if(c>1):
        print('cfl not fulfilled, c=',c)
    #else:
        #print('cfl=',c)
    u=u0(x)
    t=0
    while(t< tf):#for n in range(0,nt):
        dt=np.min([dt,tf-t])
        c=dt/dx
        un=u.copy()
        utemp0=un[0]-(c/2)*(un[1]-un[nx-1])+(c**2)/2*(un[1]-2*un[0]+un[nx-1])
       # utemp=un[nx-1]-(c/2)*(un[0]-un[nx-2])+(c**2)/2*(un[0]-2*un[nx-1]+un[nx-2])
        for i in range(1,nx): #1,...,nx-1
            u[i]=un[i]-(c/2)*(un[i+1]-un[i-1])+(c**2)/2*(un[i+1]-2*un[i]+un[i-1])            
            u[0]=utemp0
            #u[nx-1]=utemp
            u[nx]=u[0]
        if(t+dt>tf):
            print('dt resized since excedeed final time ')
            dt=tf-t 
        t=t+dt
    return u

nx=25
dt=0.04
tf=0.5
x=np.linspace(0,1,nx+1)
u=LW(nx,dt,tf)
plt.plot(x,u,'o',markerfacecolor='none')
plt.plot(x,u0(x-tf))
plt.show()



def upwind_inflow(nx,dt,tf):
    nt=int(np.ceil(tf/dt))
    x=np.linspace(0,1,nx+1)
    dx=x[2]-x[1]
    c=dt/dx    
    if(dt>dx):
        print('cfl non fulfilled, c=',c)
    else:
        print('cfl=',c)
    u=u0(x)
    t=0
    while(t< tf):
        dt=np.min([dt,tf-t])
        c=dt/dx
        un=u.copy()
        u[0]=-np.sin(2*pi*(t+dt))
        u[1]=un[1]-(dt/dx)*(un[1]+np.sin(2*pi*t)) #-sin(-2*pi*t_n)= + sin/2*pi*t_n
        for i in range(2,nx+1):
            u[i]=un[i]-(dt/dx)*(un[i]-un[i-1])
        
        if(t+dt>tf):
            print('dt resized since excedeed final time ')
            dt=tf-t 
        t=t+dt
    return u


nx=100
dt=0.01
tf=0.5
x=np.linspace(0,1,nx+1)
u=upwind_inflow(nx,dt,tf)
plt.plot(x,u,'o',markerfacecolor='none')
plt.plot(x,u0(x-tf))
plt.show()

steps=np.arange(100,400,20)
convergence_plot(steps,upwind)
convergence_plot(steps,LW)
convergence_plot(steps,upwind_inflow)
