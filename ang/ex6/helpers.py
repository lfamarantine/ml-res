import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import re
from nltk import PorterStemmer


def plotData(X, y):
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    y = y.flatten()
    pos = y==1
    neg = y==0

    # Plot Examples
    plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=10)
    plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=10)
    plt.show(block=False)


def linearKernel(x1, x2):
    #   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    #   and returns the value in sim
    # Compute the kernel
    sim = np.dot(x1, x2.T)
    return sim


def gaussianKernel(x1, x2, sigma=0.1):
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim
    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()
    # You need to return the following variables correctly.
    sim = 0
    sim = np.exp(- np.sum(np.power((x1 - x2), 2)) / float(2 * (sigma ** 2)))

    return sim


def gaussianKernelGramMatrix(X1, X2, K_function=gaussianKernel, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)
    return gram_matrix



def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    """Trains an SVM classifier"""

    y = y.flatten() # prevents warning

    # alternative to emulate mapping of 0 -> -1 in svmTrain.m
    #  but results are identical without it
    # also need to cast from unsigned int to regular int
    # otherwise, contour() in visualizeBoundary.py doesn't work as expected
    # y = y.astype("int32")
    # y[y==0] = -1

    if kernelFunction == "gaussian":
        clf = svm.SVC(C = C, kernel="precomputed", tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gaussianKernelGramMatrix(X,X, sigma=sigma), y)

    else: # works with "linear", "rbf"
        clf = svm.SVC(C = C, kernel=kernelFunction, tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(X, y)


def visualizeBoundaryLinear(X, y, model):
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    #   learned by the SVM and overlays the data on it

    # plot decision boundary
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]

    plt.plot(xp, yp, 'b-')
    plotData(X, y)


def visualizeBoundary(X, y, model, varargin=0):
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(gaussianKernelGramMatrix(this_X, X))

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors="blue", levels=[0,0])
    plt.show(block=False)


def dataset3Params(X, y, Xval, yval):
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
    #   sigma. You should complete this function to return the optimal C and
    #   sigma based on a cross-validation set.

    # You need to return the following variables correctly.
    sigma = 0.3
    C = 1
    # determining best C and sigma
    # need x1 and x2, copied from ex6.py
    x1 = [1, 2, 1]
    x2 = [0, 4, -1]

    # vector with all predictions from SVM
    predictionErrors = np.zeros((64, 3))
    predictionsCounter = 0

    # iterate over values of sigma and C
    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            print(sigma + " " + C)

            # train model on training corpus with current sigma and C
            model = svmTrain(X, y, C, "gaussian", sigma=sigma)

            # compute predictions on cross-validation set
            predictions = model.predict(gaussianKernelGramMatrix(Xval, X))

            # compute prediction errors on cross-validation set
            predictionErrors[predictionsCounter, 0] = np.mean((predictions != yval).astype(int))

            # store corresponding C and sigma
            predictionErrors[predictionsCounter, 1] = sigma
            predictionErrors[predictionsCounter, 2] = C

            # move counter up by one
            predictionsCounter = predictionsCounter + 1

    print(predictionErrors)

    # calculate mins of columns with their indexes
    row = predictionErrors.argmin(axis=0)
    m = np.zeros(row.shape)
    for i in range(len(m)):
        m[i] = predictionErrors[row[i]][i]

    # note that row[0] is the index of the min of the first column
    #   and that the first column corresponds to the error,
    #   so the row at predictionErrors(row(1),:) has best C and sigma
    print(predictionErrors[row[0], 1])
    print(predictionErrors[row[0], 2])

    # get C and sigma form such row
    sigma = predictionErrors[row[0], 1]
    C = predictionErrors[row[0], 2]

    return C, sigma


def readFile(filename):
    #   file_contents = READFILE(filename) reads a file and returns its entire
    #   contents in file_contents

    # Load File
    try:
        with open(filename, 'r') as openFile:
            file_contents = openFile.read()
    except:
        file_contents = ''
        print('Unable to open {:s}'.format(filename))

    return file_contents


def getVocabList():
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    #   and returns a cell array of the words in vocabList.

    ## Read the fixed vocabulary list
    with open('ang/ex6/data/vocab.txt', 'r') as vocabFile:
        # Store all dictionary words in dictionary vocabList
        vocabList = {}
        for line in vocabFile.readlines():
            i, word = line.split()
            vocabList[word] = int(i)

    return vocabList


def processEmail(email_contents):
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses
    #   the body of an email and returns a list of indices of the
    #   words contained in the email.

    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.find("\n\n")
    # if hdrstart:
    #     email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l = 0

    # Slightly different order from matlab version

    # Split and also get rid of any punctuation
    # regex may need further debugging...
    email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)

    for token in email_contents:

        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Stem the word
        token = PorterStemmer().stem(token.strip())

        # Skip the word if it is too short
        if len(token) < 1:
           continue

        idx = vocabList[token] if token in vocabList else 0

        # only add entries which are in vocabList
        #   i.e. those with ind ~= 0,
        #        given that ind is assigned 0 if str is not found in vocabList
        if idx > 0:
            word_indices.append(idx)

        # Print to screen, ensuring that the output lines are not too long
        if l + len(token) + 1 > 78:
            print("")
            l = 0
        print('{:s}'.format(token)),
        l = l + len(token) + 1

    print('\n\n=========================\n')

    return word_indices


def emailFeatures(word_indices):
    #   x = EMAILFEATURES(word_indices) takes in a word_indices vector and
    #   produces a feature vector from the word indices.

    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros((n, 1))

    # iterate over idx items in word_indices
    for idx in word_indices:
        # assign 1 to index idx in x
        x[idx] = 1

    return x
