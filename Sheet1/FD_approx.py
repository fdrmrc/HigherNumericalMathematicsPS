import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(math.pi*x)

def df(x):
    return math.pi*np.cos(math.pi*x)



def f(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda x: -2*np.cos(x)+3, lambda x: np.power(x,2)+1])

def df(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda x: 2*np.sin(x), lambda x: 2*x])





def error_plot(mrange,err):
    plt.figure(figsize=(10,8))
    plt.loglog(mrange,err,'-*',label='FD error')
    plt.loglog(mrange,1/np.power(mrange,1),'-',label='Order 1')
    plt.loglog(mrange,1/np.power(mrange,2),'-',label='Order 2')
    plt.loglog(mrange,1/np.power(mrange,3),'-',label='Order 3')
    plt.loglog(mrange,1/np.power(mrange,4),'-',label='Order 4')
    plt.title('error plot')
    plt.legend()
    plt.show()
    
def forward1FD(f,h):
    return (f(x+h)-f(x))/(h)

def centeredFD(f,h):
    return (f(x+h)-f(x-h))/(2*h)

def forward3FD(f,h):
    return (-11/6*f(x)+3*f(x+h)-3/2*f(x+2*h)+1/3*f(x+3*h))/(h)

def forward4FD(f,h):
    return (-25/12*f(x)+4*f(x+h)-3*f(x+2*h)+4/3*f(x+3*h)-1/4*f(x+4*h))/(h)

mrange=2**np.arange(3,14)
err=[]
an_err=[]

for M in mrange:
    x=np.linspace(-1,1,M+1)
    h=2/(M)
    e_m=np.linalg.norm(df(x)-forward1FD(f,h),np.inf)
    err.append(e_m)


for i in range(0,len(err)-1):
    an_err.append((np.log(err[i])-np.log(err[i+1]))/(np.log(mrange[i+1])-np.log(mrange[i])))

print(an_err)
error_plot(mrange,err)


err=[]
an_err=[]

for M in mrange:
    x=np.linspace(-1,1,M+1)
    h=2/(M)
    e_m=np.linalg.norm(df(x)-centeredFD(f,h),np.inf)
    err.append(e_m)


for i in range(0,len(err)-1):
    an_err.append((np.log(err[i])-np.log(err[i+1]))/(np.log(mrange[i+1])-np.log(mrange[i])))

print(an_err)
error_plot(mrange,err)


err=[]
an_err=[]

for M in mrange:
    x=np.linspace(-1,1,M+1)
    h=2/(M)
    e_m=np.linalg.norm(df(x)-forward3FD(f,h),np.inf)
    err.append(e_m)


for i in range(0,len(err)-1):
    an_err.append((np.log(err[i])-np.log(err[i+1]))/(np.log(mrange[i+1])-np.log(mrange[i])))

print(an_err)
error_plot(mrange,err)    

err=[]
an_err=[]

for M in mrange:
    x=np.linspace(-1,1,M+1)
    h=2/(M)
    e_m=np.linalg.norm(df(x)-forward4FD(f,h),np.inf)
    err.append(e_m)


for i in range(0,len(err)-1):
    an_err.append((np.log(err[i])-np.log(err[i+1]))/(np.log(mrange[i+1])-np.log(mrange[i])))

print(an_err)
error_plot(mrange,err)
