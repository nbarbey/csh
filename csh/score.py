import numpy as np
import numpy.linalg as npl
import scipy.sparse.linalg as spl
import lo

def score(A):
    """Score of a measuring matrix : sum of log of eigenvalues
    """
    if isinstance(A, lo.LinearOperator):
        B = A.todense()
    else:
        B = np.asarray(A)
    # find eigenvalues
    w, v = npl.eigh(B)
    # remove null space
    logw = np.log(w)
    logw[np.isnan(logw)] = 0.
    logw[np.isinf(logw)] = 0.
    return np.sum(logw)
