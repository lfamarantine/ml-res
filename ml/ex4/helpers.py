import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from decimal import Decimal

def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    # J = SIGMOID(z) computes the sigmoid of z.
    g = np.zeros(z.shape)
    g = expit(z)

    return g


def sigmoidGradient(z):
    #SIGMOIDGRADIENT returns the gradient of the sigmoid function
    #evaluated at z
    g = 1.0 / (1.0 + np.exp(-z))
    g = g * (1 - g)

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
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            if curr_ex > m:
                break

            # Copy the patch

            # Get the max value of the patch to normalize all examples
            max_val = max(abs(X[curr_ex - 1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))

            # Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
            # 	values directly to display_array and not to a copy of its subarray.
            # 	from stackoverflow.com/a/7960811/583834 and
            # 	bytes.com/topic/python/answers/759181-help-slicing-replacing-matrix-sections
            # Also notice the order="F" parameter on the reshape call - this is because python's
            #	default reshape function uses "C-like index order, with the last axis index
            #	changing fastest, back to the first axis index changing slowest" i.e.
            #	it first fills out the first row/the first index, then the second row, etc.
            #	matlab uses "Fortran-like index order, with the first index changing fastest,
            #	and the last index changing slowest" i.e. it first fills out the first column,
            #	then the second column, etc. This latter behaviour is what we want.
            #	Alternatively, we can keep the deault order="C" and then transpose the result
            #	from the reshape call.
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


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                        (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                        (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
    m = len(X)

    # # You need to return the following variables correctly
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # add column of ones as bias unit from input layer to second layer
    X = np.column_stack((np.ones((m, 1)), X))  # = a1

    # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    a2 = sigmoid(np.dot(X, Theta1.T))

    # add column of ones as bias unit from second layer to third layer
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

    # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    a3 = sigmoid(np.dot(a2, Theta2.T))

    # %% COST FUNCTION CALCULATION

    # % NONREGULARIZED COST FUNCTION

    # recode labels as vectors containing only values 0 or 1
    labels = y
    # set y to be matrix of size m x k
    y = np.zeros((m, num_labels))
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    for i in range(m):
        y[i, labels[i] - 1] = 1

    # at this point, both a3 and y are m x k matrices, where m is the number of inputs
    # and k is the number of hypotheses. Given that the cost function is a sum
    # over m and k, loop over m and in each loop, sum over k by doing a sum over the row

    cost = 0
    for i in range(m):
        cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))

    J = -(1.0 / m) * cost

    # % REGULARIZED COST FUNCTION
    # note that Theta1[:,1:] is necessary given that the first column corresponds to transitions
    # from the bias terms, and we are not regularizing those parameters. Thus, we get rid
    # of the first column.

    sumOfTheta1 = np.sum(np.sum(Theta1[:, 1:] ** 2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:, 1:] ** 2))

    J = J + ((lambda_reg / (2.0 * m)) * (sumOfTheta1 + sumOfTheta2))

    # %% BACKPROPAGATION

    bigDelta1 = 0
    bigDelta2 = 0

    # for each training example
    for t in range(m):

        ## step 1: perform forward pass
        # set lowercase x to the t-th row of X
        x = X[t]
        # note that uppercase X already included column of ones
        # as bias unit from input layer to second layer, so no need to add it

        # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
        a2 = sigmoid(np.dot(x, Theta1.T))

        # add column of ones as bias unit from second layer to third layer
        a2 = np.concatenate((np.array([1]), a2))
        # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
        a3 = sigmoid(np.dot(a2, Theta2.T))

        ## step 2: for each output unit k in layer 3, set delta_{k}^{(3)}
        delta3 = np.zeros((num_labels))

        # see handout for more details, but y_k indicates whether
        # the current training example belongs to class k (y_k = 1),
        # or if it belongs to a different class (y_k = 1)
        for k in range(num_labels):
            y_k = y[t, k]
            delta3[k] = a3[k] - y_k

        ## step 3: for the hidden layer l=2, set delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
        # note that we're skipping delta2_0 (=gradients of bias units, which we don't use here)
        # by doing (Theta2(:,2:end))' instead of Theta2'
        delta2 = (np.dot(Theta2[:, 1:].T, delta3).T) * sigmoidGradient(np.dot(x, Theta1.T))

        ## step 4: accumulate gradient from this example
        # accumulation
        # note that
        #   delta2.shape =
        #   x.shape      =
        #   delta3.shape =
        #   a2.shape     =
        # np.dot(delta2,x) and np.dot(delta3,a2) don't do outer product
        # could do e.g. np.dot(delta2[:,None], x[None,:])
        # seems faster to do np.outer(delta2, x)
        # solution from http://stackoverflow.com/a/22950320/583834
        bigDelta1 += np.outer(delta2, x)
        bigDelta2 += np.outer(delta3, a2)

    # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m

    # % REGULARIZATION FOR GRADIENT
    # only regularize for j >= 1, so skip the first column
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg) / m) * Theta1
    Theta2_grad += (float(lambda_reg) / m) * Theta2
    Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
    Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]

    # Unroll gradients
    grad = np.concatenate(
        (Theta1_grad.reshape(Theta1_grad.size, order='F'),
         Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad


def randInitializeWeights(L_in, L_out):
    #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    #   of a layer with L_in incoming connections and L_out outgoing
    #   connections.
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the column row of W handles the "bias" terms

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))
    # Randomly initialize the weights to small values
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init

    return W


def debugInitializeWeights(fan_out, fan_in):
    #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
    #   of a layer with fan_in incoming connections and fan_out outgoing
    #   connections using a fix set of values
    #
    #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    #   the first row of W handles the "bias" terms

    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10

    return W


def computeNumericalGradient(J, theta):
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.

    # Notes: The following code implements numerical gradient checking, and
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical
    #        approximation of) the partial derivative of J with respect to the
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    #        be the (approximately) the partial derivative of J with respect
    #        to theta(i).)
    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad



def checkNNGradients(lambda_reg=0):
    #   CHECKNNGRADIENTS(lambda_reg) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels).T

    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'),
                                Theta2.reshape(Theta2.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    # code from http://stackoverflow.com/a/27663954/583834
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similar.\n' \
             '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = Decimal(np.linalg.norm(numgrad - grad)) / Decimal(np.linalg.norm(numgrad + grad))

    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))


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
    p = np.zeros((m, 1))
    h1 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), X)), Theta1.T))
    h2 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), h1)) , Theta2.T))
    p = np.argmax(h2, axis=1)

    return p + 1
