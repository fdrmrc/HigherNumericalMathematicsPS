import numpy as np
import import_ipynb


def modthomas(a,b,c,f):
    #Thomas method, modified version
    #a=diagonal, b=subdiagonal, c=superdiagonal
    #f=rhs of the system
    
    n=len(a)
    b=np.concatenate([[0],b])
    c=np.concatenate([c,[0]])
    
    gamma=np.zeros(n)
    gamma[0]=1/a[0]
    for i in range(1,n):
        gamma[i]=(1/(a[i]-b[i]*gamma[i-1]*c[i-1]))
    
    y=np.zeros(n)
    y[0]=gamma[0]*f[0]
    for i in range(1,n):
        y[i]=gamma[i]*(f[i]-b[i]*y[i-1])
        
    x=np.zeros(n)
    x[n-1]=y[n-1]
    for i in range(n-2,-1,-1):
        x[i]=y[i]-gamma[i]*c[i]*x[i+1]

    return x


#Check 
#def tridiag(a, b, c, k1=-1, k2=0, k3=1):
#    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
#n=100
#a = -np.ones(n-1); b = 2*np.ones(n); c = np.ones(n-1)
#A = tridiag(a, b, c)
#f=np.ones(n)

#x=modthomas(b,a,c,f)
#print(x)
#print(np.linalg.solve(A,f))
