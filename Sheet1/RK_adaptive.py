import numpy as np
import matplotlib.pyplot as plt
import time
import import_ipynb

newparams = {'figure.figsize': (10, 5), 'axes.grid': True,
             'lines.linewidth': 1.5, 'lines.markersize': 5,
             'font.size': 14}
plt.rcParams.update(newparams)

def rhs(t,y):
    return np.cos(y*t**2)

def analytical(t, y):
    return 0.5*(1-np.exp(-t**2))
    

#steppers
def odesolver23(f,t,y,h):
    #Perform the next step for an ODE with rhs given by f, with step size h
    #from time t, to time t+h with expl. trapezoidal rule and a third order RK
    
    #f: rhs
    #t: current time
    #y: current solution value
    #h: current time step
    
    #Returns:
    #q: float. Order 2 approx.
    #w: float. Order 3 approx.
    
    s1 = f(t, y)
    s2 = f(t+h, y+h*s1)
    s3 = f(t+h/2.0, y+h*(s1+s2)/4.0)
    w = y + h*(s1+s2)/2.0
    q = y + h*(s1+4.0*s3+s2)/6.0
    return w, q

def odesolver12(f, t, y, h):
    s1 = f(t, y)
    s2 = f(t+h, y+h*s1)
    w = y + h*s1
    q = y + h/2.0*(s1+s2)
    return w, q
    

def rk_adaptive(ode,rhs,y0,t0,Tol,theta,tmax,iterMax=100):
    
    y=[]
    t=[]
    y.append(y0)
    t.append(t0)
    
    h=1 #step size selection: to improve (follow Hairer's book)
    i=0
    
    while (t[i]<tmax):
        w,q=ode(rhs,t[i],y[i],h)
        #local error estimate
        e=abs((w-q)/max(w,theta))
        if(e>Tol):
            h=0.8*(Tol*e)**(1/5)*h #change stepsize
            w,q=ode(rhs,t[i],y[i],h) #recompute the solution
            e=abs((w-q)/max(w,theta)) #check error again
            i2=0
            while (e>Tol) & (i2<iterMax):
                i2+=1
                h=h/2 #half the step size
                w,q=ode(rhs,t[i],y[i],h) #recompute the solution
                e=abs((w-q)/max(w,theta)) #check error
        y.append(q)
        if (t[i]+h>tmax): #hit the exact end-point
            h=tmax-t[i]
            
        t.append(t[i]+h)
        

        i+=1
        if(e<0.1*Tol):
            h=2*h
        
    
    return y,t,i


y,t,iterations=rk_adaptive(odesolver23,rhs,y0=3,t0=1,Tol=1e-4,theta=1e-3,tmax=3)


plt.plot(t,y,'-o',label='Numerical solution RK 23')
#ta=np.linspace(0,t[iterations])
#plt.plot(ta,analytical(ta,0),'-',label='Analytical solution')
plt.legend()
plt.show()
