# ------------- Machine Learning - Topic 2: Logistic Regression

# You will need to complete the following functions
# in this exercise:
#     sigmoid.py
#     costFunction.py
#     predict.py
#     plotData.py
#     plotDecisionBoundary.py

## Initialization
import numpy as np
from scipy.optimize import fmin, fmin_bfgs
import os, sys
sys.path.append(os.getcwd() + os.path.dirname('/ang/ex2/'))
from helpers import sigmoid, costFunction, predict, plotData, plotDecisionBoundary

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = np.loadtxt('ang/ex2/ex2data1.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plt, p1, p2 = plotData(X, y)

# # Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((p1, p2), ('Admitted', 'Not Admitted'), numpoints=1, handlelength=0)

plt.show(block=False) # prevents having to close the graph to move forward with ex2.py

input('Program paused. Press enter to continue.\n')
# plt.close()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
X_padded = np.column_stack((np.ones((m,1)), X))

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X_padded, y, return_grad=True)

print('Cost at initial theta (zeros): {:f}'.format(cost))
print('Gradient at initial theta (zeros):')
print(grad)

input('Program paused. Press enter to continue.\n')


## ============= Part 3: Optimizing using fmin (and fmin_bfgs)  =============
#  In this exercise, you will use a built-in function (fmin) to find the
#  optimal parameters theta.

#  Run fmin and fmin_bfgs to obtain the optimal theta
#  This function will return theta and the cost
#  fmin followed by fmin_bfgs inspired by stackoverflow.com/a/23089696/583834
#  overkill... but wanted to use fmin_bfgs, and got error if used first
myargs=(X_padded, y)
theta = fmin(costFunction, x0=initial_theta, args=myargs)
theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(costFunction, x0=theta, args=myargs, full_output=True)

# Print theta to screen
print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))
print('theta:'),
print(theta)

# Plot Boundary
plotDecisionBoundary(theta, X_padded, y)

plt.hold(False) # prevents further drawing on plot
plt.show(block=False)

input('Program paused. Press enter to continue.\n')


## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.dot(np.array([1,45,85]),theta))
print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(prob))

# Compute accuracy on our training set
p = predict(theta, X_padded)

print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))

input('Program paused. Press enter to continue.\n')
