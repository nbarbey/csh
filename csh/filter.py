"""Noise filter model"""
import numpy as np
import linear_operators as lo

def kernel_from_tod(tod, length=1000):
    """Defines Noise covariance kernel from a data set
    """
    kernel = np.zeros((tod.shape[0], np.floor(length/2) + 1))
    ftod = np.zeros((tod.shape[0], np.floor(length/2) + 1))
    # divide tod in chunks of length
    n = int(np.floor(tod.shape[1] / length))
    # compute power spectrum for each chunk and sum
    for i in xrange(n):
        iinf = i * length
        isup = (i + 1) * length
        chunk = tod[:, iinf:isup]
        ftod += np.fft.rfft(chunk, axis=1)

    kernel = np.abs(ftod) ** 2
    # zero padding
    #zero_pad = np.zeros((tod.shape[0], tod.shape[1] - kernel.shape[1]))
    #kernel = np.concatenate((kernel, zero_pad), axis=1)
    kernel = np.concatenate((kernel[:, ::-1], kernel[:, 1:]), axis=1)
    return kernel

def noise_filter(shape, kernel):
    if not isvector(kernel):
        raise ValueError('Expected a vector shaped kernel')
    kern = kernel.flatten()
    if kernel.shape[0] < shape[1]:
        kern = np.zeros(shape[0])
        kern[:kernel.shape[0]] = kern
    if kernel.shape[0] > shape[1]:
        kern = kernel[:shape[0]]
    F = lo.fftn(shape, axes=(0,))
    K = lo.axis_mul(shape, kern, axis=0, dtype=np.complex128)
    return F.T * K * F

def kernel_convolve(shapein, kernel):
    from copy import copy
    shapeout = list(copy(shapein))
    #shapeout[1] = shapein[1] + kernel.size - 1
    #shapeout[1] = shapein[1] - kernel.size + 1
    def matvec(x):
        y = np.zeros(shapeout)
        for i in xrange(x.shape[0]):
            y[i] = np.convolve(x[i], kernel, mode='same')
        return y
    def rmatvec(x):
        y = np.zeros(shapein)
        for i in xrange(x.shape[0]):
            y[i] = np.convolve(x[i], kernel, mode='same')
        return y
    return lo.ndoperator(shapein, shapeout, matvec, rmatvec, dtype=kernel.dtype)

def kernels_convolve(shapein, kernel):
    from copy import copy
    shapeout = list(copy(shapein))
    #shapeout[1] = shapein[1] + kernel.size - 1
    #shapeout[1] = shapein[1] - kernel.size + 1
    def matvec(x):
        y = np.zeros(shapeout)
        for i in xrange(x.shape[0]):
            y[i] = np.convolve(x[i], kernel[i], mode='same')
        return y
    def rmatvec(x):
        y = np.zeros(shapein)
        for i in xrange(x.shape[0]):
            y[i] = np.convolve(x[i], kernel[i], mode='same')
        return y
    return lo.ndoperator(shapein, shapeout, matvec, rmatvec, dtype=kernel.dtype)

def isvector(arr):
    if arr.ndim == 1:
        return True
    elif arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return True
    else:
        return False
