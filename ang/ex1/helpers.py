

def computeCost(X, y, theta):
    m = len(y)
    J = 1 / (2 * m) * sum(np.dot(X, theta.T) - y)**2
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    jh = np.zeros(num_iters)
    for i in range(num_iters):
        h = theta[0] + theta[1] * X[:, 1]
        th_0 = theta[0] - alpha * (1 / m * sum(h - y))
        th_1 = theta[1] - alpha * (1 / m * sum(h - y))
        theta = np.array([th_0, th_1])
        jh[i] = computeCost(X=X, y=y, theta=theta)

    return theta



