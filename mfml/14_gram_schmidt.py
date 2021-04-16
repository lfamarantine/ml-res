import numpy as np
import numpy.linalg as la


def gram_schmidt_process(x):
    B = np.array(x, dtype=np.float_)
    for i in range(B.shape[1]):
        for j in range(i):
            # note: @ is dot-product
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        if la.norm(B[:, i]) > 1e-10:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])
    return B


def reflect_matrix(x):
    e = gram_schmidt_process(x=x)
    te = np.identity(x.shape[1])
    # mirror operates by negating the last component of a matrix
    te[-1, -1] = -1
    # transformation matrix
    t = e @ te @ la.inv(e)
    return t


# use case: reflecting an object (i.e. picture)
# - calculate orthonormal basis of a matrix
m = np.array([[1, 0, 2, 6],
              [0, 1, 8, 2],
              [2, 8, 3, 1],
              [1, -6, 2, 3]], dtype=np.float_)
m_orthonormal = gram_schmidt_process(x=m)
# - calculate transformation matrix of a reflection of matrix m
T = reflect_matrix(x=m)



