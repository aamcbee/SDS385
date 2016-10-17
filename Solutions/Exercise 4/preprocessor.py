import numpy as np
from sklearn.datasets import load_svmlight_files

Filenames = ['Day'+str(x)+'.svm' for x in xrange(121)]
# Load in the svmlight files using sklearn.
# This creates a list of X and y arrays for each day.
# Note that each X array is stored as a csr sparse matrix
# This process takes about 14-15 minutes
Xy = load_svmlight_files(Filenames)

# Rewrite the successes to be 0 or 1, extract the y's from the list of X and y's
y = (np.hstack([Xy[i] for i in np.arange(1, len(Xy) + 1, 2)]) > 0).astype(float)

# Extract all of the X's from the Xy list
X = [Xy[i] for i in np.arange(0, len(Xy), 2)]
Data = []
Index = []

# Process the sparse matrices by extracting the dense indices for each sample and their
# Corresponding data.
for Xj in X:
    for i in np.arange(Xj.shape[0]):
        Data.append(np.hstack((1,Xj[i].data))) # Note we add 1 to get an intercept term
        Index.append(np.hstack((0,Xj[i].indices + 1))) # Note we shift the indices to get the intercept term

# Save the list of sample data, the list of sample indices, and the y values
# in binary file form
np.save('Data', Data)
np.save('Index', Index)
np.save('Yvals', y)
