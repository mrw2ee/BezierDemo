import numpy as np
import math

def bbasis(x:list, n:int) -> np.ndarray:
    """
    Compute Bernstein basis vectors at query points x for n-th order polynomial

    Inputs:
        x - array_like with m elements
        n - integer

    Output:
        B - m x (n+1) matrix, or n+1 vector (when m==1), where m=size(x)
    """
    if np.isscalar(x):
        x = np.array(x)
    else:
        x = np.reshape(x,(-1,1))
    
    X = x**range(0,n+1)
    Xcomp = (1-x)**np.array(range(n,-1,-1))
    coef = math.factorial(n)/np.array([math.factorial(x)*math.factorial(n-x) for x in range(0,n+1)])
    return coef*(X*Xcomp)


def bbasisderivative(x:list, n:int) -> np.ndarray:
    """
    Compute derivative of the Bernstein basis vectors at query points x
    """
    Btmp = bbasis(x,n-1)
    if len(Btmp.shape)==1:
        z = 0.
    else:
        z = np.zeros([Btmp.shape[0],1]) 
    tmp = np.block([z,Btmp,z])

    return -n*np.diff(tmp)

