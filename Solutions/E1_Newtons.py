
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


# In[79]:

def NM_inverse(X, y, beta, m, iters=1000, tol=1e-3):
    '''
    This function implements newton's method on the negative log likelihood using
    simple inversion of the Hessian matrix (likely to be unstable)
    
        X: An m x n array of features
    beta: An n vector of the current "solution" to the system
    y: An m vector of the number of successes, where y_i is the integer number of success for 
        a given trial. (Note that for our wdbc data, this is only 1 or 0)
    m: An m a vector of the number of trials (where m_i is the number of trials for the ith case)
        (Note that for our wdbc data, this is simply a vector of ones)
    iters: An optional int indicating the maximum number of iterations to implement Newton's method
    
    '''
    ll = []
    betas = []
    beta0 = beta
    for i in np.arange(iters):
        ll.append(negloglike(X,y,beta0,m))
        betas.append(beta0)
        
        beta1 = beta0 - np.dot(la.inv(grad2_lb(X, y, beta0, m)), grad_lb(X,y,beta0,m))
        llb0 = negloglike(X,y,beta0,m)
        llb1 = negloglike(X,y,beta1,m)
        
        # Here we measure convergence based on the absolute difference of l(beta0) and l(beta1)
        if np.abs(llb0 - llb1) > tol:
            beta0 = beta1
        else:
            ll.append(negloglike(X,y,beta0,m))
            betas.append(beta1)
            return beta1, ll, betas, i
    
    return beta1, ll, betas, i


# In[80]:

# Load in columns 3-12 of the wdbc data
X = np.loadtxt('wdbc.csv', dtype='float', delimiter=',', usecols=np.arange(3,13))
X1 = scale(X, axis=1) # Scale X
X = np.hstack((np.ones((X.shape[0],1)), X1))
cl = np.loadtxt('wdbc.csv', dtype='str', delimiter=',', usecols=[1])

# Set Malignant to be 1 and Benign to be 0
y = (cl == 'M').astype(float)
M = np.ones(y.shape[0])
m,n = X.shape

# Select 3 random starting vectors
b1 = np.random.rand(n)
b2 = np.random.rand(n) 
b3 = np.random.rand(n) 

# Plot the convergence of the log likelihood for various starting beta
for i in [b1,b2,b3, np.ones(n), np.zeros(n)]:
    beta0, ll, betas, i = NM_inverse(X,y,i,M, tol=1e-5)
    plt.plot(ll)

plt.title("Convergence of Newton's Method for various starting beta")
plt.xlabel("Number of iterations")
plt.ylabel("Negative Log-likelihood")
plt.show()


# In[ ]:



