# ------------- Machine Learning - Topic 4: Neural Networks Learning

#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions
#  in this exercise:
#
#     displayData.py
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py

from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
import os, sys
sys.path.append(os.getcwd() + os.path.dirname('/ang/ex4/'))
from helpers import displayData, nnCostFunction, sigmoidGradient, randInitializeWeights, checkNNGradients, predict
## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = loadmat('ang/ex4/ex4data1.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# crucial step in getting good performance!
# changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten()

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

displayData(sel)

input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
mat = loadmat('ang/ex4/ex4weights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

# Unroll parameters
# ndarray.flatten() always creates copy (http://stackoverflow.com/a/28930580/583834)
# ndarray.ravel() requires transpose to have matlab unrolling order (http://stackoverflow.com/a/15988852/583834)
# np.append() always makes a copy (http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.append.html)
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# # Weight regularization parameter (we set this to 0 here).
lambda_reg = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

print('Training Set Accuracy: {:f}\n(this value should be about 0.287629)'.format(J))

input('Program paused. Press enter to continue.\n')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('Checking Cost Function (w/ Regularization)...')

# Weight regularization parameter (we set this to 1 here).
lambda_reg = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

print('Cost at parameters (loaded from ex4weights): {:f}\n(this value should be about 0.383770)'.format(J))

input('Program paused. Press enter to continue.\n')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('Evaluating sigmoid gradient...')

g = sigmoidGradient( np.array([1, -0.5, 0, 0.5, 1]) )
print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]:')
print(g)

input('Program paused. Press enter to continue.\n')


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('Initializing Neural Network Parameters...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'),
                                    initial_Theta2.reshape(initial_Theta2.size, order='F')))

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('Checking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()

input('Program paused. Press enter to continue.\n')


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambda_reg = 3
checkNNGradients(lambda_reg)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

print('\n\nCost at (fixed) debugging parameters (w/ lambda_reg = 3): {:f} ' \
         '\n(this value should be about 0.576051)\n\n'.format(debug_J))

input('Program paused. Press enter to continue.\n')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('Training Neural Network...')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
#  You should also try different values of lambda_reg
#  note that scipy.optimize.minimize() can use a few different solver
#   methods for gradient descent:
#   http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(nnCostFunction, x0=nn_params, args=myargs,
                   options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                 (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                 (num_labels, hidden_layer_size + 1), order='F')

input('Program paused. Press enter to continue.\n')


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:, 1:])

input('Program paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y) * 100)))







