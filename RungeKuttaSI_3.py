import numpy as np
import math as m
import matplotlib.pyplot as plt
#from scipy.optimize import newton_krylov
%matplotlib inline


#***TO DO:
#Implement a modified Newton's method (constant Jacobian as option)


def my_Newton(f,df,x0,tol,maxiter):
    itercount=0
    if(np.size(f(x0))==1):
        delta=-f(x0)/df(x0)
        while (np.linalg.norm(delta)>tol and itercount<maxiter):
            itercount+=1
            x0=x0+delta
            delta=-f(x0)/df(x0)
        x0=x0+delta
    else:
        delta=np.linalg.solve(df(x0),-f(x0))
        while (np.linalg.norm(delta,np.Inf)>tol and itercount<maxiter):
            itercount+=1
            x0=x0+delta
            delta=np.linalg.solve(df(x0),-f(x0))
        x0=x0+delta

    if (itercount>=maxiter):
        print('Massimo numero di iterazioni eseguito:', (iter))

    return x0

def modified_Newton(f,dfapprox,x0,tol,maxiter):
    itercount=0
    if(np.size(f(x0))==1):
        delta=-f(x0)/dfapprox
        while (np.linalg.norm(delta)>tol and itercount<maxiter):
            itercount+=1
            x0=x0+delta
            delta=-f(x0)/dfapprox
        x0=x0+delta
    else:
        delta=np.linalg.solve(dfapprox,-f(x0))
        while (np.linalg.norm(delta,np.Inf)>tol and itercount<maxiter):
            itercount+=1
            x0=x0+delta
            delta=np.linalg.solve(dfapprox,-f(x0))
        x0=x0+delta

    if (itercount>=maxiter):
        print('Massimo numero di iterazioni eseguito:', (iter))

    return x0





def RKSI3_step(rhs,y,t,h,drhs,):
    #compute a step of RK method with:
    #f: RHS
    #y: current solution
    #h: current step size
    #t: current time
    #drhs: analyitically compute Jacobian of the RHS

    #return: numerical solution at time t+h

    #Butcher's tableau elements definition
    a11=(3+m.sqrt(3))/6
    a12=0
    a21=-m.sqrt(3)/3
    a22=a11
    b1=0.5
    b2=b1
    c1=(3+m.sqrt(3))/6
    c2=(3-m.sqrt(3))/6

    Tol=.01*h**3 #local error
    maxiter=100
    x0=y #guess starting point is current solution. Usually good guess
    def F(x): return x-y-a11*h*rhs(t+c1*h,x)
    JFapprox= np.eye(np.size(y))-a11*h*drhs(t+0.5*(c1+c2)*h,y)
    #def JF(x): return np.eye(np.size(y))-a11*h*drhs(t+c1*h,x)
    s1=modified_Newton(F,JFapprox,x0,Tol,maxiter)
    #s1=newton_krylov(F,x0)

    def F(x): return x- y -a21*h*rhs(t+h*c1,s1)-a22*h*rhs(t+h*c2,x)
    #def JF(x): return np.eye(np.size(y)) - a22*h*drhs(t+h*c2,x)
    #s2=newton_krylov(F,x0)
    s2=modified_Newton(F,JFapprox,x0,Tol,maxiter)

    return y+h*(0.5*rhs(t+h*c1,s1)+0.5*rhs(t+h*c2,s2))


def rhs(t,x):
    return np.array([-x[0],x[1]])
def drhs(t,x):
    return np.array([[-1,0],[0,1]])

tf=1
ts=50
h=tf/ts
y=np.zeros([2,ts])
y0=np.array([1,1])
y[:,0]=y0

t=0
for i in range(0,ts-1):
    y[:,i+1]=RKSI3_step(rhs,y[:,i],t,h,drhs)
    t=t+h

t=np.linspace(0,tf,ts)
plt.plot(t,y[0,:],'*',label='numerical',)
plt.plot(t,np.exp(-t),'-',label='analytical')
plt.legend()
plt.show()


#Convergence check
err=[]
tsrange=np.arange(25,400,25)
for ts in tsrange:
    h=tf/ts
    y=np.zeros([2,ts])
    y[:,0]=y0

    t=0
    for i in range(0,ts-1):
        y[:,i+1]=RKSI3_step(rhs,y[:,i],t,h,drhs)
        t=t+h

    err.append(np.linalg.norm(np.array([np.exp(-t),np.exp(t)])-y[:,-1],np.Inf))



plt.loglog(tsrange,err,'*',tsrange,err[-1]*np.power((tsrange/tsrange[-1]),-3),'-')
plt.title('Convergence plot')
plt.show()
