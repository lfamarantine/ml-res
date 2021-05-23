import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def linearRegCostFunction(X, y, theta, lambda_val, return_grad=False):
    #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda_val) computes the
    #   cost of using theta as the parameter for linear regression to fit the
    #   data points in X and y. Returns the cost in J and the gradient in grad

    # Initialize some useful values
    m = len(y) # number of training examples

    # force to be 2D vector
    theta = np.reshape(theta, (-1,y.shape[1]))

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    # cost function
    J = (1. / (2 * m)) * np.power((np.dot(X, theta) - y) , 2).sum() + (float(lambda_val) / (2*m)) * np.power(theta[1:theta.shape[0]],2).sum()

    # regularized gradient
    grad = (1. / m) * np.dot(X.T, np.dot(X,theta) - y) + (float(lambda_val) / m) * theta

    # unregularize first gradient
    grad_no_regularization = (1. / m) * np.dot(X.T, np.dot(X, theta) - y)
    grad[0] = grad_no_regularization[0]

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J




def trainLinearReg(X, y, lambda_val):
    #   [theta] = TRAINLINEARREG (X, y, lambda_val) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda_val. Returns the
    #   trained parameters theta.

    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))

    # Short hand for cost function to be minimized
    def costFunc(theta):
        return linearRegCostFunction(X, y, theta, lambda_val, True)

    # Now, costFunction is a function that takes in only one argument
    maxiter = 200
    results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

    theta = results["x"]

    return theta




def learningCurve(X, y, Xval, yval, lambda_val):
    #   [error_train, error_val] = ...
    #       LEARNINGCURVE(X, y, Xval, yval, lambda_val) returns the train and
    #       cross validation set errors for a learning curve. In particular,
    #       it returns two vectors of the same length - error_train and
    #       error_val. Then, error_train(i) contains the training error for
    #       i examples (and similarly for error_val(i)).
    #
    #   In this function, you will compute the train and test errors for
    #   dataset sizes from 1 up to m. In practice, when working with larger
    #   datasets, you might want to do this in larger intervals.

    # Number of training examples
    m = len(X)

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(1, m + 1):
        # define training variables for this loop
        X_train = X[:i]
        y_train = y[:i]

        # learn theta parameters with current X_train and y_train
        theta = trainLinearReg(X_train, y_train, lambda_val)

        # fill in error_train(i) and error_val(i)
        #   note that for error computation, we set lambda_val = 0 in the last argument
        error_train[i - 1] = linearRegCostFunction(X_train, y_train, theta, 0)
        error_val[i - 1] = linearRegCostFunction(Xval, yval, theta, 0)

    return error_train, error_val


def polyFeatures(X, p):
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #

    # You need to return the following variables correctly.
    # X_poly = np.zeros((X.size, p))

    # initialize X_poly to be equal to the single-column X
    X_poly = X

    # if p is equal or greater than 2
    if p >= 2:

        # for each number between column 2 (index 1) and last column
        for k in range(1, p):
            # add k-th column of polynomial features where k-th column is X.^k
            X_poly = np.column_stack((X_poly, np.power(X, k + 1)))

    return X_poly


def featureNormalize(X):
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma



def plotFit(min_x, max_x, mu, sigma, theta, p):
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure
    plt.hold(True)

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)) # 1D vector

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)

    # Hold off to the current figure
    plt.hold(False)


def validationCurve(X, y, Xval, yval):
    #   [lambda_vec, error_train, error_val] = ...
    #       VALIDATIONCURVE(X, y, Xval, yval) returns the train
    #       and validation errors (in error_train, error_val)
    #       for different values of lambda. You are given the training set (X,
    #       y) and validation set (Xval, yval).

    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(len(lambda_vec)):
        lambda_val = lambda_vec[i]
        # learn theta parameters with current lambda value
        theta = trainLinearReg(X, y, lambda_val)
        # fill in error_train[i] and error_val[i]
        #   note that for error computation, we set lambda = 0 in the last argument
        error_train[i] = linearRegCostFunction(X, y, theta, 0)
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)

    return lambda_vec, error_train, error_val




