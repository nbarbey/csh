"""Define compression modes"""
import numpy as np
from copy import copy
import linear_operators as lo

def identity(tod, factor):
    shape = tod.shape
    return lo.identity(2 * (np.prod(shape),), dtype=np.float64)

def averaging(tod, factor, dtype=np.float64):
    """Averaging compression mode"""
    shape = tod.shape
    nsamples = tod.nsamples
    shapes = [(tod.shape[0], n) for n in nsamples]
    Bs = [lo.binning(s, factor=factor, axis=1, dtype=dtype) for s in shapes]
    B = lo.concatenate(Bs, axis=1)
    S = lo.ndhomothetic(shape, 1. / factor, dtype=dtype)
    return B * S

def decimate_temporal(tod, factor):
    shape = tod.shape
    mask = np.ones(shape, dtype=bool)
    mask[:, 0::factor] = 0.
    return lo.decimate(mask)

def cs(tod, factor):
    """ Compressed sensing compression mode"""
    # shape
    shape = tod.shape
    # transform
    H = lo.fht(shape, axes=0)
    # mask
    start_ind = np.resize(np.arange(factor), shape[0])
    mask = np.ones(shape, dtype=bool)
    for i in xrange(mask.shape[0]):
        mask[i, start_ind[i]::factor] = 0.
    M = lo.decimate(mask, dtype=np.float64)
    C = M * H
    return C

def binning3d(tod, factors):
    shape = tod.shape
    if len(shape) is not 3:
        raise ValueError('Expected 3d shape')
    B0 = lo.binning(shape, factor=factors[0], axis=0, dtype=np.float64)
    shape1 = np.asarray(copy(shape))
    shape1[0] /= factors[0]
    B1 = lo.binning(shape1, factor=factors[1], axis=1, dtype=np.float64)
    shape2 = np.asarray(copy(shape1))
    shape2[1] /= factors[1]
    B2 = lo.binning(shape2, factor=factors[2], axis=2, dtype=np.float64)
    return B2 * B1 * B0

class AnyOperator(object):
    """
    Use any linearoperator as a compression scheme.
    """
    def __init__(self, mat):
        self.mat = lo.aslinearoperator(mat)
    def __call__(self, tod, factor=None):
        shape = tod.shape
        if factor is not None:
            print("Warning : compression factor defined by compression matrix")
        factor = self.mat.shape[1] / self.mat.shape[0]
        n_repeats = np.prod(shape) / self.mat.shape[1]
        return lo.interface.block_diagonal(n_repeats * (self.mat,))

