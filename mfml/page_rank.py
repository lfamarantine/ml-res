import numpy as np
import numpy.linalg as la


def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    # damping mechanism for procrastinating elements (to combat self-referencing)
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    # assuming 100 agents clicking on a single website at a time
    r = 100 * np.ones(n) / n
    lastR = r
    r = M @ r
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r
    return r


L2 = np.array([[0.5,  0,  0,  0,  0.2],
               [0,  0,  0,  0.5,  0.2],
               [0,  1,  0,  0.5,  0.2],
               [0.5,  0,  0,  0,  0.2],
               [0,  0,  1,  0,  0.2]])

pageRank(L2, 1)


