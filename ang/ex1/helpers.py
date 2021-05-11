import numpy as np
import matplotlib.pyplot as plt


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


def gradientDescent(X, y, theta, alpha, num_iters):

    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = computeCost(X, y, theta)

    return theta


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



def featureNormalize(X):

    # FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
    	mu[:,i] = np.mean(X[:,i])
    	sigma[:,i] = np.std(X[:,i])
    	X_norm[:,i] = (X[:,i] - float(mu[:,i]))/float(sigma[:,i])

    return X_norm, mu, sigma


def gradientDescentMulti(X, y, theta, alpha, num_iters):

    # GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta = theta - alpha*(1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history


def normalEqn(X, y):
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

    return theta