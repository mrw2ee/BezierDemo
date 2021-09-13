import numpy as np
import bernstein

def test_bbasis_x():
    """
    bbasis should accept both maxrix and scalar inputs
    """
    
    # Scalar
    x = 0.5
    B = bernstein.bbasis(x,2)

    # Matrix
    x = np.array([[0.3,0.5],[1,1]])
    B = bernstein.bbasis(x,2)

def test_bbasis_sum1():
    """
    At each query point [0,1] the basis vectors should sum to one regardless polynomial order.
    """
    x = np.linspace(0,1,100)
    for n in range(1,10):
        B = bernstein.bbasis(x,n)
        np.testing.assert_allclose(np.sum(B,axis=1), 1, atol=1e-15)



def test_bbasisderivative():
    # Verify analytic derivative of basis vectors
    #
    # Each Bernstin basis function is a scalar-valued function of a single variable.
    # However, we compute the derivatives of all Basis functions simultaneously for
    # multiple querry points. We select an initial vector query points and
    # numerically compute the derivative for all basis functions at multiple step
    # sizes.

    # Initial query point
    x0 = np.linspace(0,1,100)
    # Basis order
    n = 5
    fctn = lambda x: bernstein.bbasis(x,n)
    # Analytic derivatives at x0 (matrix - not a Jacobian)
    g = bernstein.bbasisderivative(x0,n)
    # Initial function values (vector)
    f0 = fctn(x0)

    # Step size
    Nsteps = 50
    #e0=1e-9
    #eps_all = e0*np.logspace(0,10,num=Nsteps)
    e0=1e-7
    eps_all = e0*np.logspace(0,4,num=Nsteps)
    err = np.zeros((len(x0),Nsteps))
    for s in range(Nsteps):
        eps = eps_all[s]
        # Numeric gradient
        ghat = (fctn(x0 + eps) -f0)/eps
        err[:,s] = np.linalg.norm(np.abs(ghat-g),axis=1)

    assert max(np.var(err/eps_all,axis=1))<1e-3,"Error does not grow linearly with slope epsilon"
