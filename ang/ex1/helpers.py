import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname('ang/ex1/'))
import computeCost as cc


def gradientDescent(X, y, theta, alpha, num_iters):

    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = cc.computeCost(X, y, theta)

    return theta


def computeCost(X, y, theta):

    #COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples
    J = 0
	#	theta is an (n+1)-dimensional vector
	#	X is an m x (n+1)-dimensional matrix
	#	y is an m-dimensional vector
    s = (X.dot(theta) - np.transpose([y]))**2
    J = (1.0 / (2 * m)) * s.sum(axis=0)

    return J


def plotData(x, y):
    # PLOTDATA Plots the data points x and y into a new figure
    #   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    #   population and profit.
    plt.plot(x, y, 'rx', markersize=10, label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show(block=False)  # prevents having to close the chart


def warmUpExercise(*args, **kwargs):
    return np.identity(5)
