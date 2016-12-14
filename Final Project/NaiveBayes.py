import numpy as np


class GaussNaiveBayes(object):
    '''
    This class performs Gaussian Naive Bayes classifications
    '''
    def __init__(self, labels):
        '''
        Initialize the Naive Bayes classifier.
        
        Inputs:
            labels: a list of possible labels for the dataset
        
        Outputs:
            GaussNaiveBayes object
        '''
        
        self.labels = labels
        self.n_labels = len(labels)

    def fit(self, X, y):
        '''
        Fit the parameters of the model to the training data
        X: an N x P array, where N is the number of training samples and P is the number of 
           features
        Y: an N array of labels for each sample
        '''
        N, P = X.shape
        # Set the number of features for the classifier.
        self.n_features = P
        
        # First, split the data according to each label
        # Initialize a list of indices corresponding to each label
        self.lab_ind = []
        for i in self.labels:
            # Append the boolean mask of indices for which samples correspond to the given label
            self.lab_ind.append((y==i))
                
        self.means = np.zeros((self.n_labels, P))
        self.var = np.zeros((self.n_labels, P))
        self.label_probs = np.zeros(self.n_labels)
        for i in xrange(self.n_labels):
            self.means[i] = X[self.lab_ind[i]].mean(axis=0)
            self.var[i] = X[self.lab_ind[i]].var(axis=0)
            # Calculate the proportion of samples with each label in the training set
            self.label_probs[i] = (self.lab_ind[i]).sum() / float(N)
    
    def predict(self, X):
        '''
        Predict the labels for an array of test samples.
        
        Inputs:
            X: an M x P array, where M is the number of testing samples and P is the number of 
                features        
        Outputs:
            yguess: an M array of guesses for the label values for each test sample.
        '''
        M, P = X.shape
        # Make sure the test set matches number of features of the fitted classifier.
        assert P == self.n_features, str(P) + " != " + str(self.n_features) + ", the number of features in the test set does not match the number of features in the classifer."
        
        # Calculate the log of the gaussian probability for each sample given the sample mean and sample variance
        # for each possible feature, for each label. 
        # Note that we use array broadcasting to perform all of these operations at once, and add an extra dimension to
        # X to allow us to broadcast each sample to the self.means and self.var arrays.
        loggauss = np.log(1. / np.sqrt(2 * np.pi * self.var)) + (-1 * (X[:,None] - self.means)**2 / (2. * self.var))
        # Use the independence assumption to sum over the feature probabilities for each label (sum over the 3rd axis)
        probability = np.log(1. / self.n_labels) + np.sum(loggauss, axis=2)
        yguess = np.argmax(probability, axis=1)
        labelguess = np.array(self.labels)[yguess]
        return labelguess
        
    def accuracy(self, X, y):
		'''
		Return the accuracy of the classifier for predicting the labels of a set
		of test samples (here, the labels of the test samples are already known)

		Inputs:
			X: an M x P array, where M is the number of testing samples and P is the 
				number of features
			y: a P array of labels corresponding to each sample in X.
			
		Outputs:
			acc: float, the percentage of correctly predicted labels for the test
				sample.
		'''
		M, P = X.shape
		yguess = self.predict(X)
		acc = (yguess == y).sum() / float(M)
		return acc
