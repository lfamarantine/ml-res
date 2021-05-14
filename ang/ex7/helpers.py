import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.linalg as linalg


def displayData(X, example_width=None):
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # using plt.ion() instead of the commented section below
    # # closes previously opened figure. preventing a
    # # warning after opening too many figures
    # plt.close()

    # # creates new figure
    # plt.figure()

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


def findClosestCentroids(X, centroids):
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    #   in idx for a dataset X where each row is a single example. idx = m x 1
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    # from http://stackoverflow.com/a/24261734/583834
    # to avoid error "arrays used as indices must be of integer (or boolean) type"
    idx = np.zeros((X.shape[0], 1), dtype=np.int8)
    # set m = # of training examples
    m = X.shape[0]

    # for every training example
    for i in range(m):

        # for every centroid
        for j in range(K):

            # compute the euclidean distance between the example and the centroid
            difference = X[i, :] - centroids[j, :]
            distance = np.power(np.sqrt(difference.dot(difference.T)), 2)
            # if this is the first centroid, initialize the min_distance and min_centroid
            # OR
            # if distance < min_distance, reassign min_distance=distance and min_centroid to current j
            if j == 0 or distance < min_distance:
              min_distance = distance
              min_centroid = j

        # assign centroid for this example to one corresponding to the min_distance
        idx[i]= min_centroid

    return idx


def computeCentroids(X, idx, K):
    #   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
    #   computing the means of the data points assigned to each centroid. It is
    #   given a dataset X where each row is a single data point, a vector
    #   idx of centroid assignments (i.e. each entry in range [1..K]) for each
    #   example, and K, the number of centroids. You should return a matrix
    #   centroids, where each row of centroids is the mean of the data points
    #   assigned to it.

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    # for each centroid
    for j in range(K):
        # find training example indices that are assigned to current centroid
        # notice the [0] indexing - it's necessary because of np.nonzero()'s
        #   two-array output
        centroid_examples = np.nonzero(idx == j)[0]

        # compute mean over all such training examples and reassign centroid
        centroids[j, :] = np.mean(X[centroid_examples, :], axis=0)

    return centroids


def hsv(n=63):
    return colors.hsv_to_rgb( np.column_stack([np.linspace(0, 1, n+1), np.ones(((n+1), 2))]))


def drawLine(p1, p2, **kwargs):
    #   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    #   current figure
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def plotDataPoints(X, idx, K):
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    #   with the same index assignments in idx have the same color
    # Create palette (see hsv.py)
    palette = hsv(K)
    colors = np.array([palette[int(i)] for i in idx])
    # Plot the data
    plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=colors)

    return


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.

    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=400, c='k', linewidth=1)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :], c='b')

    # Title
    plt.title('Iteration number {:d}'.format(i+1))

    return


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a true/false flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to false by default. runkMeans returns
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set default value for plot progress
    # (commented out due to pythonic default parameter assignment above)
    # if not plot_progress:
    #     plot_progress = False

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # if plotting, set up the space for interactive graphs
    if plot_progress:
        plt.close()
        plt.ion()

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        sys.stdout.write('\rK-Means iteration {:d}/{:d}...'.format(i + 1, max_iters))
        sys.stdout.flush()

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.')

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print('\n')

    return centroids, idx


def kMeansInitCentroids(X, K):
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X

    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    return centroids


def featureNormalize(X):
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma


def pca(X):
    #   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    #   Returns the eigenvectors U, the eigenvalues (on diagonal) in S

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # compute the covariance matrix
    sigma = (1.0/m) * (X.T).dot(X)
    U, S, Vh = linalg.svd(sigma)
    S = linalg.diagsvd(S, len(S), len(S))

    return U, S


def projectData(X, U, K):
    #   Z = projectData(X, U, K) computes the projection of
    #   the normalized inputs X into the reduced dimensional space spanned by
    #   the first K columns of U. It returns the projected examples in Z.

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # get Z - the projections from X onto the space defined by U_reduce
    #	note that this vectorized version performs the projection the instructions
    # 	above but in one operation
    Z = X.dot(U_reduce)

    return Z


def recoverData(Z, U, K):
    #   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
    #   original data that has been reduced to K dimensions. It returns the
    #   approximate reconstruction in X_rec.

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # recover data
    X_rec = Z.dot(U_reduce.T)

    return X_rec


