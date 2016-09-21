
# coding: utf-8

# In[8]:

import numpy as np
from scipy.optimize import fmin_cg
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.stats import binom
import scipy.misc as misc
from sklearn.preprocessing import scale


# In[29]:

def grad_lb(X, y, beta, m):
    '''
    This function calculates the gradient of the negative log-likelihood for the
    given values.
    Here, we assume the regression model y_i ~ Binomial(m_i, w_i), where y_i is an 
    integer number of "successes", m_i is the number of trials for the ith case, 
    and w_i is a regression on a feature vector x_i given by the inverse logit transform
    
    X: An m x n array of features
    beta: An n vector of the current "solution" to the system
    y: An m vector of the number of successes, where y_i is the integer number of success for 
        a given trial. (Note that for our wdbc data, this is only 1 or 0)
    m: An m a vector of the number of trials (where m_i is the number of trials for the ith case)
        (Note that for our wdbc data, this is simply a vector of ones)
    '''
    
    Xbeta = np.dot(X, beta)
    # Takes the logistic function of Xbeta (inverse logit transform)
    w = expit(Xbeta)
    # Calculate m_i * w_i - y_i using broadcasting
    mw_yX = (m * w - y).reshape(y.shape[0],1) * X
    # sum (m_iw_i - y_i)X over the rows
    gradlb = mw_yX.sum(axis=0)
    return gradlb


# In[30]:

def grad2_lb(X, y, beta, m):
    '''
    This function calculates the gradient^2 of the negative log-likelihood for the
    given values.
    As before, we assume the regression model y_i ~ Binomial(m_i, w_i), where y_i is an 
    integer number of "successes", m_i is the number of trials for the ith case, 
    and w_i is a regression on a feature vector x_i given by the inverse logit transform
    
    X: An m x n array of features
    beta: An n vector of the current "solution" to the system
    y: An m vector of the number of successes, where y_i is the integer number of success for 
        a given trial. (Note that for our wdbc data, this is only 1 or 0)
    m: An m a vector of the number of trials (where m_i is the number of trials for the ith case)
        (Note that for our wdbc data, this is simply a vector of ones)
    '''
    N, P = X.shape
    Xbeta = np.dot(X,beta)
    # Takes the logistic function of Xbeta (inverse logit transform)
    w = expit(Xbeta)
    # Calculate m_iw_i(1-w_i) as the vector of scalar matrix "coefficients" for our sum
    # Here we use simple broadcasting
    coeff = m * w * (1. - w)
    grad2 = np.zeros((P,P))
    for i in np.arange(N):
        grad2 += coeff[i] * np.outer(X[i], X[i])
    return grad2


# In[31]:

def negloglike(X, y, beta, m):
    '''
    This function computes the negative log likelihood given features X, successes y,
    trials m and solution vector beta.
    '''
    Xbeta = np.dot(X, beta)
    w = expit(Xbeta)
    # Use scipy's binomial object to compute the log of the pmf, then take the sum   
    return -binom.logpmf(y, m, w).sum()





