import math
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize


def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    # J = SIGMOID(z) computes the sigmoid of z.
    g = np.zeros(z.shape)
    g = expit(z)

    return g


def displayData(X, example_width=None):
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # closes previously opened figure. preventing a
    # warning after opening too many figures
    plt.close()

    # creates new figure
    plt.figure()

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))

    # Set example_width automatically if not passed in
    if not example_width or not 'example_width' in locals():
        example_width = int(round(math.sqrt(X.shape[1])))

    # Gray Image
    plt.set_cmap("gray")

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((int(pad + display_rows * (example_height + pad)), int(pad + display_cols * (example_width + pad))))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            if curr_ex > m:
                break

            # Get the max value of the patch to normalize all examples
            max_val = max(abs(X[curr_ex - 1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))
            display_array[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = np.reshape(X[curr_ex - 1, :],
                                                                                   (example_height, example_width),
                                                                                   order="F") / max_val
            curr_ex += 1

        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    # Do not show axis
    plt.axis('off')
    plt.show(block=False)

    return h, display_array


def lrCostFunction(theta, X, y, lambda_reg, return_grad=False):
    # LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    # regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda_reg) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # taken from costFunctionReg.py
    one = y * np.transpose(np.log(sigmoid(np.dot(X,theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X,theta))))
    reg = (float(lambda_reg) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = -(1. / m) * (one + two).sum() + reg
    grad = (1./m) * np.dot(sigmoid( np.dot(X,theta) ).T - y, X).T + (float(lambda_reg) / m)*theta
    # the case of j = 0 (recall that grad is a n+1 vector)
    grad_no_regularization = (1. / m) * np.dot(sigmoid( np.dot(X, theta) ).T - y, X).T
    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    # display cost at each iteration
    sys.stdout.write("Cost: %f   \r" % (J) )
    sys.stdout.flush()

    if return_grad:
        return J, grad.flatten()
    else:
        return J


def oneVsAll(X, y, num_labels, lambda_reg):
    # ONEVSALL trains multiple logistic regression classifiers and returns all
    # the classifiers in a matrix all_theta, where the i-th row of all_theta
    # corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    for c in range(num_labels):
        # initial theta for c/class
        initial_theta = np.zeros((n + 1, 1))

        print("Training {:d} out of {:d} categories...".format(c + 1, num_labels))

        ## functions WITH gradient/jac parameter
        myargs = (X, (y % 10 == c).astype(int), lambda_reg, True)
        theta = minimize(lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 13},
                         method="Newton-CG", jac=True)

        # assign row of all_theta corresponding to current c/class
        all_theta[c, :] = theta["x"]

    return all_theta


def predictOneVsAll(all_theta, X):
    # PREDICT Predict the label for a trained one-vs-all classifier. The labels
    # are in the range 1..K, where K = size(all_theta, 1).
    #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #  for 4 examples)

    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))
    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))
    p = np.argmax(sigmoid( np.dot(X,all_theta.T) ), axis=1)

    return p


def predict(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m,1))

    # add column of ones as bias unit from input layer to second layer
    X = np.column_stack((np.ones((m,1)), X)) # = a1

    # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    a2 = s.sigmoid( np.dot(X,Theta1.T) )

    # add column of ones as bias unit from second layer to third layer
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))

    # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    a3 = s.sigmoid( np.dot(a2,Theta2.T) )

    # get indices as in predictOneVsAll
    p = np.argmax(a3, axis=1)

    return p + 1

