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
    lam: Float, the penalty coefficient for the l2 penalty

    ----------
    Outputs:
    ----------
    b0: P vector: The final solved for vector for the sample segment that
        "minimizes" the negative log-likelihood up to that point.
        Note that this is a view of the updated vector b0.
    '''

    # Initialize various values
    # N is the number of samples in the block, alpha and C correspond to the robbin's monro stepsize
    N = len(Data)
    C = .5
    alpha = .75
    # Set all the stepsizes for the block, beginning with the step-start value
    stepsize = C * np.arange(step_start + 1, step_start + N + 1)**(-alpha)
    
    # Calculate all of the nonzero indices used in the samples as a whole
    B_ind = np.zeros_like(b0)
    for I in Index:
        B_ind[I] = 1.
    bindex = B_ind.nonzero()
    
    # Run the SGD algorithm on the block
    for j in xrange(N):
        # Extract the appropriate sample
        Xj = Data[j] 
        # Extract the sparse index for the specific sample
        sp_index = Index[j] 
        # Extract the b0 values only on the sparse index for this sample (for gradient calculation)
        b1 = b0[sp_index] 
        # Take care of penalty terms on bindex
        b0[bindex] *= (1 - 2 * lam * stepsize[j]) 
        b0[sp_index] -= stepsize[j] * gi(Xj, y[j], m[j], b1)
    # Compensate for penalty terms on trivial sample indices
    b0 *= (1 - 2 * lam * stepsize.sum())
    b0[bindex] /= (1 - 2 * lam * stepsize.sum())
    return b0

@jit
def sp_rm_sgd(Data, Index, y, m, b0, lam=1e-5, blocksize=1000):
    '''
    This implements the stochastic gradient descent algorithm for logistic regression
    using Robbin's Monro stepsize choice for the entire sparse dataset. Note that the
    algorithm is implemented in chunks of size "blocksize," which allows us to use
    rm_sgd_segment on more reasonable chunks of the data.

    Inputs:
    ----------
    Note that these inputs are based on a sparse dataset of N samples with P features.

    Data: A list of N arrays, each corresponding to the non-trivial data for each
        sample. This list (along with Index and y) is generated for the url dataset
        using preprocessor.py
    Index: A list of N arrays, each corresponding to the non-trivial indices for each
        sample.
    Y: An N-vector that records the label, or "number of successes" for each sample.
        This array is typically comprised of zeros and ones.
    m: An N-vector that records the number of trials per sample. This is typically an
        array of all ones.
    b0: A P-vector, the initial guess for the logistic regression coefficients. Typically
        we begin with a vector of all zeros. Note that this vector is consumed during the
        operation (operated on as a view)
    lam: A float, the penalty coefficient
    blocksize: An integer, the size of the blocks on which to operate with rm_sgd_segment

    Returns:
    ------------
    b0: The final vector of regression coefficients
    '''
    # Find the number of samples and calculate the number of even splits that can be made
    # for the given blocksize.
    N = len(Data)
    splits = int(N / blocksize)

    # Carry out the SGD algorithm on each subsequent chunk of data
    for i in np.arange(splits):
        # Get the appropriate indices for this chunk of data
        j = blocksize * i
        k = blocksize * (i + 1)
        # Update the guess vector on the chunk of data using rm_sgd_segment
        b0 = rm_sgd_segment(Data[j:k], Index[j:k], y[j:k], m[j:k], b0, j, lam=lam)

    # This compensates for any "hangover" should the blocksize not evenly divide N.
    j = splits * blocksize
    b0 = rm_sgd_segment(Data[j:N], Index[j:N], y[j:N], m[j:N], b0, j, lam=lam)
    return b0

def predict(D, I, y, b0):
    '''
    This function makes binomial predictions for a set of samples and returns
    the accuracy of the predictions.
    Inputs:
    ---------
    D: A list of the N nontrivial data arrays for each testing sample respectivly
    I: A list of the N index arrays corresponding to the nontrivial data for each sample
        respectively.
    y: An N array of the actual number of "successes per sample"
    b0: A P array of regression coefficients

    Returns:
    --------
    Accuracy: A float value of the the percentage of correct predictions, on a 0-100 scale
    '''
    n = len(D)
    Predict = np.zeros(n)
    for i in xrange(n):
        Predict[i] = expit(np.dot(D[i], b0[I[i]]))
    Predict = (Predict > .5).astype(float)
    Accuracy = (Predict == y).sum() / float(n) * 100
    return Accuracy
