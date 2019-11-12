import numpy as np



def thomas_symmetric(a,b,f):
    #a = diagonal
    #b = super/sub diagonal (matrix is symmetric)
    
    N = len(f)  #f_0,...f_n-1
    c= np.zeros(N)
    c[0] = a[0]
    for i in range(1,N):#da 1 a N-1
        c[i] = a[i] - (b[i-1]**2)/c[i-1]
    
    g = np.zeros(N)
    g[0] = f[0]
    for j in range(1,N):
        g[j] = f[j] - (b[j-1]/c[j-1])*g[j-1]
    
    u = np.zeros(N)
    u[N-1] = g[N-1]/c[N-1]
    for k in range(N-2,-1,-1):
        u[k] = (g[k]-b[k]*u[k+1])/c[k]

    return u











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
