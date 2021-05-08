import numpy as np
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

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))

        # ============================================================

        # Save the cost J in every iteration
        J_history[i] = cc.computeCost(X, y, theta)

    return theta