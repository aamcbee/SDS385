import numpy as np
from scipy.special import expit
from scipy.stats import binom
from matplotlib import pyplot as plt
from scipy import linalg as la

def gi(xi,yi,mi,bi):
    '''
    Computes the individual gradient contribution for the negative log-likelihood

    xi: P vector corresponding to the ith sample
    yi: Float corresponding to the ith sample's number of successes
    mi: Float corresponding to the ith sample's number of trials
    bi: P vector corresponding to the current guess vector
    '''
    w = expit(np.dot(xi,bi))
    return (mi*w - yi) * xi

def nll(xi, yi, mi, bi):
    '''
    Computes the individual negative log likelihood contribution for a given
    sample.
    xi: P vector corresponding to the ith sample
    yi: Float corresponding to the ith sample's number of successes
    mi: Float corresponding to the ith sample's number of trials
    bi: P vector corresponding to the current guess vector
    '''
    w = expit(np.dot(xi,bi))
    # Note we add  or subtract 1e-16 to eliminate an infinite log likelihood
    if np.abs(w) < 1e-16:
        w = 1e-16
    if np.abs(w - 1) < 1e-16:
        w = 1-1e-16
    return -yi * np.log(w) - (mi - yi) * np.log(1. - w)

def full_nll(X, y, m, b):
    '''
    Computes the full negative log likelihood for a set of samples X
    X: N x P array of samples
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b: P vector corresponding to the current guess vector.
    '''
    w = expit(np.dot(X, b))
    return -( y * np.log(w+1e-10) + (m-y) * np.log(1 - w + 1e-10)).sum()

def full_g(X, y, m, b):
    w = expit(np.dot(X,b))
    mw_yX = (m * w - y).reshape(y.shape[0],1) * X
    return mw_yX.sum(axis=0)

def ls(x, p, f, gf, c=.5, scale=.75, iters=100):
    '''
    Uses the backward line search algorithm to calculate an appropriate stepsize for the algorithm
    minimizing function f.

    x: P vector: the input vector
    p: P vector: the descent direction
    f: function to be minimized, takes in x as single argument
    gf: gradient of f, takes in x as a single argument

    Returns:
    -------------
    a: The stepsize calculated
    A boolean to indicate whether the algorithm converged or not
    '''
    fx = f(x)
    #gfxp = (gf(x) * p).sum()
    gfxp = np.dot(gf(x), p)
    a = 1.
    for i in np.arange(iters):
        if f(x + a * p) <= fx + c * a * gfxp:
            return a, True
        else:
            a *= scale
    return a, False

def simple_sgd(X, y, m, b0, iters=10, tol=1e-3, stepsize=.001, check_full=False, smooth=.05):
    '''
    Runs the stochastic gradient descent algorithm for the negative log likelihood
    function of a given dataset using the robbin's-monro stepsize calculation and exponentially
    weighted moving average convergence criterion.

    Inputs:
    ----------
    X: N x P array of samples
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b0: P vector corresponding to the initial guess vector
    iters: integer, optional argument set to maximum number of times to run through
        the sample set X, set to 10
    tol: float, optional argument for the error bound used to determine convergence.
    check_full: boolean, optional argument to determine whether to calculate the full
        negative log likelihood at each step
    ----------
    Outputs:
    ----------
    b1: P vector: The final solved for vector that "minimizes" the negative log-likelihood
    betas: A list of the beta values stored at each iteration.
    ll: A list of the exponentially weighted moving average of the nll contribution at each
        iteration
    full: If check_full == True, then this is a list of the full negative log likelihood at
        each iteration. If not, this simply returns an empty set

    '''

    # Initialize various values
    ll = []
    full = []
    betas = []
    N, P = X.shape
    index = np.arange(N)
    C = .5
    alpha = .75

    for i in np.arange(iters):
        np.random.shuffle(index)
        for j in np.arange(N):
            k = index[j]

            # This ensures the first computed nll contribution is in place
            if i==0 and j==0:
                ll.append(nll(X[k], y[k], m[k], b0))

            # This computes the full log-likelihood for the entire dataset at the
            # b point
            if check_full:
                full.append(full_nll(X, y, m, b0))

            betas.append(b0)
            b1 = b0 - stepsize * gi(X[k], y[k], m[k], b0)
            llb1 = smooth * nll(X[k], y[k], m[k], b1) + (1-smooth) * ll[-1]
            ll.append(llb1)

            # This allows for a burn in period because convergence is measured using the EWMA
            if j < N * .01:
                b0 =b1
            else:
                if np.abs(ll[-1] - ll[-2]) > tol:
                    b0 = b1
                else:
                    betas.append(b1)
                    return b1, betas, ll, full
    betas.append(b1)
    return b1, betas, ll, full


def rm_sgd(X, y, m, b0, iters=10, tol=1e-3, check_full=False, smooth=.05):
    '''
    Runs the stochastic gradient descent algorithm for the negative log likelihood
    function of a given dataset using the robbin's-monro stepsize calculation and exponentially
    weighted moving average convergence criterion.

    Inputs:
    ----------
    X: N x P array of samples
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b0: P vector corresponding to the initial guess vector
    iters: integer, optional argument set to maximum number of times to run through
        the sample set X, set to 10
    tol: float, optional argument for the error bound used to determine convergence.
    check_full: boolean, optional argument to determine whether to calculate the full
        negative log likelihood at each step
    ----------
    Outputs:
    ----------
    b1: P vector: The final solved for vector that "minimizes" the negative log-likelihood
    betas: A list of the beta values stored at each iteration.
    ll: A list of the exponentially weighted moving average of the nll contribution at each
        iteration
    full: If check_full == True, then this is a list of the full negative log likelihood at
        each iteration. If not, this simply returns an empty set

    '''

    # Initialize various values
    ll = []
    full = []
    betas = []
    N, P = X.shape
    index = np.arange(N)
    C = .5
    alpha = .75

    for i in np.arange(iters):
        np.random.shuffle(index)
        for j in np.arange(N):
            stepsize = C * (j + 1)**(-alpha)
            k = index[j]

            # This ensures the first computed nll contribution is in place
            if i==0 and j==0:
                ll.append(nll(X[k], y[k], m[k], b0))

            # This computes the full log-likelihood for the entire dataset at the
            # b point
            if check_full:
                full.append(full_nll(X, y, m, b0))

            betas.append(b0)
            b1 = b0 - stepsize * gi(X[k], y[k], m[k], b0)
            llb1 = smooth * nll(X[k], y[k], m[k], b1) + (1-smooth) * ll[-1]
            ll.append(llb1)

            # This allows for a burn in period because convergence is measured using the EWMA
            if j < N * .01:
                b0 =b1
            else:
                if np.abs(ll[-1] - ll[-2]) > tol:
                    b0 = b1
                else:
                    betas.append(b1)
                    return b1, betas, ll, full
    betas.append(b1)
    return b1, betas, ll, full

def ls_sgd(X, y, m, b0, iters=10, tol=1e-3, check_full=False, smooth=.05, batchsize=50):
    '''
    Runs the stochastic gradient descent algorithm for the negative log likelihood
    function of a given dataset using the minibatch line search method and exponentially
    weighted moving average convergence criterion.

    Inputs:
    ----------
    X: N x P array of samples
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b0: P vector corresponding to the initial guess vector
    iters: integer, optional argument set to maximum number of times to run through
        the sample set X, set to 10
    tol: float, optional argument for the error bound used to determine convergence.
    check_full: boolean, optional argument to determine whether to calculate the full
        negative log likelihood at each step
    batchsize: integer, optional argument specifying the size of the minibatch used to
        calculate stepsize.
    ----------
    Outputs:
    ----------
    b1: P vector: The final solved for vector that "minimizes" the negative log-likelihood
    betas: A list of the beta values stored at each iteration.
    ll: A list of the exponentially weighted moving average of the nll contribution at each
        iteration
    full: If check_full == True, then this is a list of the full negative log likelihood at
        each iteration. If not, this simply returns an empty set

    '''

    # Initialize various values
    ll = []
    full = []
    betas = []
    N, P = X.shape
    index = np.arange(N)
    C = .5
    alpha = .75

    for i in np.arange(iters):
        np.random.shuffle(index)
        for j in np.arange(N):

            k = index[j]
            if j % 50 == 0: # Criterion to calculate a new stepsize
                batch_index = np.random.choice(index, size=batchsize, replace=False)
                Xbatch = X[batch_index]
                ybatch = y[batch_index]
                mbatch = m[batch_index]
                p = -full_g(Xbatch, ybatch, mbatch, b0) / batchsize

                stepsize, suff = ls(b0, p, lambda x: full_nll(Xbatch, ybatch, mbatch, x), lambda x:full_g(Xbatch, ybatch, mbatch, x))
                # If the line search method, in essence, didn't work, use robbins monro step size
                if stepsize == 1. or not suff:
                    stepsize = .5 * (j + 1)**(-.75)

            # This ensures the first computed nll contribution is in place
            if i==0 and j==0:
                ll.append(nll(X[k], y[k], m[k], b0))

            # This computes the full log-likelihood for the entire dataset at the
            # b point
            if check_full:
                full.append(full_nll(X, y, m, b0))

            betas.append(b0)
            b1 = b0 - stepsize * gi(X[k], y[k], m[k], b0)
            llb1 = smooth * nll(X[k], y[k], m[k], b1) + (1-smooth) * ll[-1]
            ll.append(llb1)

            # This allows for a burn in period because convergence is measured using the EWMA
            if j < N * .01:
                b0 =b1
            else:
                if np.abs(ll[-1] - ll[-2]) > tol:
                    b0 = b1
                else:
                    betas.append(b1)
                    return b1, betas, ll, full
    betas.append(b1)
    return b1, betas, ll, full

def adagrad(X, y, m, b0, iters=10, tol=1e-3, check_full=False, smooth=.05, stepsize = .01):
    '''
    Runs the stochastic gradient descent algorithm for the negative log likelihood
    function of a given dataset using the adagrad method and exponentially
    weighted moving average convergence criterion.

    Inputs:
    ----------
    X: N x P array of samples
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b0: P vector corresponding to the initial guess vector
    iters: integer, optional argument set to maximum number of times to run through
        the sample set X, set to 10
    tol: float, optional argument for the error bound used to determine convergence.
    check_full: boolean, optional argument to determine whether to calculate the full
        negative log likelihood at each step
    ----------
    Outputs:
    ----------
    b1: P vector: The final solved for vector that "minimizes" the negative log-likelihood
    betas: A list of the beta values stored at each iteration.
    ll: A list of the exponentially weighted moving average of the nll contribution at each
        iteration
    full: If check_full == True, then this is a list of the full negative log likelihood at
        each iteration. If not, this simply returns an empty set

    '''

    # Initialize various values
    ll = []
    full = []
    betas = []
    N, P = X.shape
    index = np.arange(N)
    G = np.zeros(P) # Diagonal of G, "hessian approximation"
    for i in np.arange(iters):
        np.random.shuffle(index)
        for j in np.arange(N):
            k = index[j]

            # This ensures the first computed nll contribution is in place
            if i==0 and j==0:
                ll.append(nll(X[k], y[k], m[k], b0))

            # This computes the full log-likelihood for the entire dataset at the
            # b point
            if check_full:
                full.append(full_nll(X, y, m, b0))

            betas.append(b0)
            
            # Calculate the gradient contribution
            grad_i = gi(X[k], y[k], m[k], b0)
            # Update G
            G += grad_i**2
            # Compute the next beta iteration
            b1 = b0 - stepsize * np.sqrt(G)**(-1) * grad_i
            llb1 = smooth * nll(X[k], y[k], m[k], b1) + (1-smooth) * ll[-1]
            ll.append(llb1)

            # This allows for a burn in period because convergence is measured using the EWMA
            if j < N * .01:
                b0 =b1
            else:
                if np.abs(ll[-1] - ll[-2]) > tol:
                    b0 = b1
                else:
                    betas.append(b1)
                    return b1, betas, ll, full
    betas.append(b1)
    return b1, betas, ll, full
