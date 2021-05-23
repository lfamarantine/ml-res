import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def estimateGaussian(X):
    #   [mu sigma2] = estimateGaussian(X),
    #   The input X is the dataset with each n-dimensional data point in one row
    #   The output is an n-dimensional vector mu, the mean of the data set
    #   and the variances sigma^2, an n x 1 vector

    m, n = X.shape
    # You should return these values correctly
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    # estimating mu - at this point it is an n-column vector
    # mu = (1/m)*sum(X,1);
    mu = np.mean(X, axis=0)
    # turn into n-rows vector
    mu = mu.T

    # estimating sigma^2 = std.dev.
    # normalizes with 1/N, instead of with 1/(N-1) in formula std (x) = 1/(N-1) SUM_i (x(i) - mean(x))^2
    # i.e. degrees of freedom = 0 (by default)
    sigma2 = np.var(X, axis=0)

    # turn into n-rows vector
    sigma2 = sigma2.T

    return mu, sigma2


def multivariateGaussian(X, mu, sigma2):
    #    p = MULTIVARIATEGAUSSIAN(X, mu, sigma2) Computes the probability
    #    density function of the examples X under the multivariate gaussian
    #    distribution with parameters mu and sigma2. If sigma2 is a matrix, it is
    #    treated as the covariance matrix. If sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    k = len(mu)

    # turns 1D array into 2D array
    if sigma2.ndim == 1:
        sigma2 = np.reshape(sigma2, (-1,sigma2.shape[0]))

    if sigma2.shape[1] == 1 or sigma2.shape[0] == 1:
        sigma2 = linalg.diagsvd(sigma2.flatten(), len(sigma2.flatten()), len(sigma2.flatten()))

    # mu is unrolled (and transposed) here
    X = X - mu.reshape(mu.size, order='F').T

    p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma2), -0.5) ) * \
        np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))

    return p



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
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
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


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_var):
    #   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    #   num_features, lambda) returns the cost and gradient for the
    #   collaborative filtering problem.

    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features), order='F')
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features), order='F')

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # X * Theta performed according to low rank matrix vectorization
    squared_error = np.power(np.dot(X,Theta.T) - Y,2)

    # for cost function, sum only i,j for which R(i,j)=1
    J = (1/2.) * np.sum(squared_error * R)

    # NOTE where filtering through R is applied:
    # 	at ( (X * Theta') - Y ) - NOT after multiplying by Theta
    # 	that means that for purposes of the gradient, we're only interested
    # 	in the errors/differences in i, j for which R(i,j)=1
    # NOTE also that even though we do a sum, we only do it over users,
    # 	so we still get a matrix
    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta)
    # NOTE also that even though we do a sum, we only do it over movies,
    # 	so we still get a matrix
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X)

    ### COST FUNCTION WITH REGULARIZATION
    # only add regularized cost to J now
    J = J + (lambda_var / 2.) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

    ### GRADIENTS WITH REGULARIZATION
    # only add regularization terms
    X_grad = X_grad + lambda_var*X
    Theta_grad = Theta_grad + lambda_var*Theta

    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))

    return J, grad


def checkCostFunction(lambda_var=0):
    #   CHECKCOSTFUNCTION(lambda_var) Creates a collaborative filering problem
    #   to check your cost function and gradients, it will output the
    #   analytical gradients produced by your code and the numerical gradients
    #   (computed using computeNumericalGradient). These two gradient
    #   computations should result in very similar values.

    # Set lambda_var
    # if not lambda_var or not 'lambda_var' in locals():
    #     lambda_var = 0

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return cofiCostFunc(p, Y, R, num_users, num_movies, num_features, lambda_var)

    numgrad = computeNumericalGradient(costFunc, params)

    cost, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_var)


    print(np.column_stack((numgrad, grad)))
    print('The above two columns you get should be very similar.\n' \
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). ' \
             '\nRelative Difference: {:e}'.format(diff))



def selectThreshold(yval, pval):
    #   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    #   threshold to use for selecting outliers based on the results from a
    #   validation set (pval) and the ground truth (yval).

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        # make anomaly predictions based on which values of pval are less than epsilon
        #   i.e. which examples from the cross-validation set have a very low p(x)
        cvPredictions = pval < epsilon

        # calculate F1 score, starting by calculating true positives(tp)
        # false positives (fp) and false negatives (fn)

        # true positives is the intersection between
        #   our positive predictions (cvPredictions==1) and positive ground truth values (yval==1)
        tp = np.sum(np.logical_and((cvPredictions == 1), (yval == 1)).astype(float))

        # false positives are the ones we predicted to be true (cvPredictions==1) but weren't (yval==0)
        fp = np.sum(np.logical_and((cvPredictions == 1), (yval == 0)).astype(float))

        # false negatives are the ones we said were false (cvPredictions==0) but which were true (yval==1)
        fn = np.sum(np.logical_and((cvPredictions == 0), (yval == 1)).astype(float))

        # compute precision, recall and F1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


def visualizeFit(X, mu, sigma2):
    #   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
    #   probability density function of the Gaussian distribution. Each example
    #   has a location (x1, x2) that depends on its feature values.

    X1,X2 = np.meshgrid(np.arange(0, 35.1, 0.5), np.arange(0, 35.1, 0.5))
    Z = multivariateGaussian(np.column_stack((X1.reshape(X1.size, order='F'), X2.reshape(X2.size, order='F'))), mu, sigma2)
    Z = Z.reshape(X1.shape, order='F')

    plt.plot(X[:, 0], X[:, 1],'bx', markersize=13, markeredgewidth=1)

    plt.hold(True)
    # Do not plot if there are infinities
    if (np.sum(np.isinf(Z)) == 0):
        plt.contour(X1, X2, Z, np.power(10,(np.arange(-20, 0.1, 3)).T))

    plt.hold(False)


def loadMovieList():
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
    #   and returns a cell array of the words in movieList.

    ## Read the fixed movieulary list
    with open("ml/ex8/data/movie_ids.txt") as movie_ids_file:

        # Store all movies in movie list
        n = 1682  # Total number of movies

        movieList = [None]*n
        for i, line in enumerate(movie_ids_file.readlines()):
            movieName = line.split()[1:]
            movieList[i] = " ".join(movieName)

    return movieList


def normalizeRatings(Y, R):
    #   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    #   has a rating of 0 on average, and returns the mean rating in Ymean.

    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean



