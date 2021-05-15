# ------------- Machine Learning - Topic 8: Anomaly Detection and Recommender Systems

#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions
#  in this exercise:
#
#     estimateGaussian
#     multivariateGaussian
#     visualizeFit
#     selectThreshold

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.getcwd() + os.path.dirname('/ang/ex8/'))
from helpers import estimateGaussian, multivariateGaussian, visualizeFit, selectThreshold


## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

print('Visualizing example dataset for outlier detection.\n');

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = loadmat('ang/ex8/data/ex8data1.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx', markersize=10, markeredgewidth=1)
plt.axis([0,30,0,30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show(block=False)

input('Program paused. Press enter to continue.')


## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution,
#  then compute the probabilities for each of the points and then visualize
#  both the overall distribution and where each of the points falls in
#  terms of that distribution.
#
print('Visualizing Gaussian fit.\n')

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)

#  Returns the density of the multivariate normal at each data point (row)
#  of X
p = multivariateGaussian(X, mu, sigma2)

#  Visualize the fit
plt.close()
visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show(block=False)

input('Program paused. Press enter to continue.')

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
#

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: {:e}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)\n')

# Find the outliers in the training set and plot the
outliers = p < epsilon

# interactive graphs
plt.ion()

#  Draw a red circle around those outliers
plt.hold(True)
# plt.scatter(X[outliers, 0], X[outliers, 1], s=325, facecolors='none', edgecolors='r')
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=18, fillstyle='none', markeredgewidth=1)
plt.hold(False)
plt.show(block=False)

input('Program paused. Press enter to continue.')

## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a
#  harder problem in which more features describe each datapoint and only
#  some features indicate whether a point is an outlier.
#

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = loadmat('ang/ex8/data/ex8data2.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

#  Training set
p = multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {:e}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))
print('# Outliers found: {:d}'.format(np.sum((p < epsilon).astype(int))))
input('   (you should see a value epsilon of about 1.38e-18)\n')







