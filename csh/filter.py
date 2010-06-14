"""Noise filter model"""
import numpy as np
import lo

def kernel_from_tod(tod, length=1000):
    """Defines Noise covariance kernel from a data set
    """
    kernel = np.zeros((tod.shape[0], np.floor(length/2) + 1))
    # divide tod in chunks of length
    n = np.floor(tod.shape[1] / length)
    print n
    # compute power spectrum for each chunk and sum
    for i in xrange(n):
        iinf = i * length
        isup = (i + 1) * length
        chunk = tod[:, iinf:isup]
        ftod = np.fft.rfft(chunk, axis=1)
        kernel += np.abs(ftod) ** 2
    # zero padding
    zero_pad = np.zeros((tod.shape[0], tod.shape[1] - kernel.shape[1]))
    kernel = np.concatenate((kernel, zero_pad), axis=1)
    return kernel

def noise_filter(shape, kernel):
    F = lo.fftn(shape, axes=(0,))
    K = lo.axis_mul(shape, kernel, axis=1, dtype=np.complex128)
    return F.T * K * F
