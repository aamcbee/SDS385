import numpy as np
from scipy.special import expit
from scipy.stats import binom
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse as sp
from numba import jit

#@jit(nopython=True)
def gi(xi,yi,mi,bi, lam=.25):
    '''
    Computes the individual gradient contribution for the negative log-likelihood

    xi: P vector corresponding to the ith sample
    yi: Float corresponding to the ith sample's number of successes
    mi: Float corresponding to the ith sample's number of trials
    bi: P vector corresponding to the current guess vector
    '''
    w = expit(np.dot(xi, bi))
    return (mi*w - yi) * xi


#@jit(nopython=True)
def rm_sgd_segment(Data, Index, y, m, b0, step_start, lam=.05):
    '''
    Runs the stochastic gradient descent algorithm for the negative log likelihood
    function of a given dataset using the robbin's-monro stepsize calculation and exponentially
    weighted moving average convergence criterion on a given segment of samples.

    Inputs:
    ----------
    Data: A list of nontrivial data arrays for each sample. These data arrays are
        extracted from a sparse matrix row using X[i].data
    Index: A list of the indices corresponding to the nontrivial data for each sample.
    y: N vector of the number of successes per sample
    m: N vector of the number of trials per sample
    b0: P vector corresponding to the initial guess vector. This argument is consumed
        throughout the algorithm.
    stepstart: Int, the starting step for the segment
    lam: Float, the penalty constant for the l2 penalty

    ----------
    Outputs:
    ----------
    b0: P vector: The final solved for vector for the sample segment that
        "minimizes" the negative log-likelihood up to that point.
        Note that this is a view of the updated vector b0.
    '''

    # Initialize various values
    #N, P = X.shape
    N = len(Data)
    C = .5
    alpha = .75
    # SPLIT UP DATA, CHANGE THE STEPSIZE ACCORDINGLY
    stepsize = C * np.arange(step_start + 1, step_start + N+1)**(-alpha)

    B_ind = np.zeros_like(b0)
    for I in Index:
        B_ind[I] = 1.
    bindex = B_ind.nonzero()
    for j in xrange(N):
        Xj = Data[j]
        sp_index = Index[j]
        b1 = b0[sp_index]
        b0[bindex] *= (1 - 2 * lam * stepsize[j])
        b0[sp_index] -= stepsize[j] * gi(Xj, y[j], m[j], b1)
    b0 *= (1 - 2 * lam * stepsize.sum())
    b0[bindex] /= (1 - 2 * lam * stepsize.sum())
    return b0
