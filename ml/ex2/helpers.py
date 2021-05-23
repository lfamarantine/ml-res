from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    # J = SIGMOID(z) computes the sigmoid of z.
    g = np.zeros(z.shape)
    g = expit(z)

    return g

def costFunction(theta, X, y, return_grad=False):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y) # number of training examples
    J = 0
    grad = np.zeros(theta.shape)
    one = y * np.transpose(np.log(sigmoid(np.dot(X, theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X, theta))))
    J = -(1. / m) * (one + two).sum()
    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T

    if return_grad == True:
        return J, np.transpose(grad)
    elif return_grad == False:
        return J


def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    m = X.shape[0] # Number of training examples
    p = np.zeros((m, 1))
    sigValue = sigmoid(np.dot(X,theta))
    p = sigValue >= 0.5

    return p



def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.plot(X[pos,0], X[pos,1], marker='+', markersize=9, color='k')[0]
    p2 = plt.plot(X[neg,0], X[neg,1], marker='o', markersize=7, color='y')[0]

    return plt, p1, p2


def mapFeature(X1, X2):
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #   for a total of 1 + 2 + ... + (degree+1) = ((degree+1) * (degree+2)) / 2 columns
    #
    #   Inputs X1, X2 must be the same size
    degree = 6
    out = np.ones(( X1.shape[0], sum(range(degree + 2)) ))
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i+1):
            out[:, curr_column] = np.power(X1, i-j) * np.power(X2, j)
            curr_column += 1

    return out


def plotDecisionBoundary(theta, X, y):
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones
    fig = plt.figure()
    fig, p1, p2 = plotData(X[:, 1:3], y)
    fig.hold(True)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
        # Plot, and adjust axes for better viewing
        p3 = fig.plot(plot_x, plot_y)
        # Legend, specific for the exercise
        fig.legend((p1, p2, p3[0]), ('Admitted', 'Not Admitted', 'Decision Boundary'), numpoints=1, handlelength=0.5)
        fig.axis([30, 100, 30, 100])
        fig.show(block=False)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
        z = np.transpose(z)  # important to transpose z before calling contour
        # Notice you need to specify the level 0
        # we get collections[0] so that we can display a legend properly
        p3 = fig.contour(u, v, z, levels=[0], linewidth=2).collections[0]
        # Legend, specific for the exercise
        fig.legend((p1, p2, p3), ('y = 1', 'y = 0', 'Decision Boundary'), numpoints=1, handlelength=0)
        fig.show(block=False)

    fig.hold(False)  # prevents further drawing on plot


def costFunctionReg(theta, X, y, lambda_reg, return_grad=False):
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y) # number of training examples
    J = 0
    grad = np.zeros(theta.shape)
    one = y * np.transpose(np.log(sigmoid( np.dot(X,theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid( np.dot(X,theta))))
    reg = (float(lambda_reg) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = -(1. / m)*(one + two).sum() + reg
    # applies to j = 1,2,...,n - NOT to j = 0
    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta) ).T - y, X).T + (float(lambda_reg) / m)*theta
    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1. / m) * np.dot(sigmoid( np.dot(X, theta)).T - y, X).T
    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J

