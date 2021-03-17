import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ang/ex1/ex1data1.txt', names=['x', 'y'])

# ----- Descriptive analysis
X = data.loc[:, 'x'].values.reshape(-1)
y = data.loc[:, 'y'].values

plt.scatter(x=X, y=y)
plt.ylabel('Population of City in 10.000s')
plt.xlabel('Profit in $10.000s')
plt.show()


# ----- Cost & Gradient descent
# add intercept
X = np.column_stack((np.ones(len(X)), X))
iterations = 1000
alpha = 0.01
# sample cost based on arbitrary theta
J = computeCost(X=X, y=y, theta=np.zeros(2))
J = computeCost(X=X, y=y, theta=np.array([-1, 2]))
# fitted theta
theta = gradientDescent(X=X, y=y, theta=np.array([-1, 2]), alpha=alpha, num_iters=iterations)




