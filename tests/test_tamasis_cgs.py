#!/usr/bin/env python
import numpy as np
import scipy.sparse.linalg as spl
import tamasis as tm
import lo

# data
pacs = tm.PacsObservation(filename=tm.tamasis_dir+'tests/frames_blue.fits')
tod = pacs.get_tod()
# projector
projection = tm.Projection(pacs, resolution=3.2, 
                           oversampling=True, npixels_per_sample=6)
masking = tm.Masking(tod.mask)
compression = tm.CompressionAverage(pacs.compression_factor)
model = masking * compression * projection
# naive map
naive = model.transpose(tod)
# coverage map
coverage = model.transpose(tod.ones(tod.shape))
# noise covariance
length = 2**np.ceil(np.log2(np.array(tod.nsamples) + 200))
invNtt = tm.InvNtt(length, pacs.get_filter_uncorrelated())
fft = tm.Fft(length)
padding = tm.Padding(left=invNtt.ncorrelations, 
                     right=length - tod.nsamples - invNtt.ncorrelations)
weight = padding.T * fft.T * invNtt * fft * padding
W = lo.aslinearoperator(weight.aslinearoperator())
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator())
# priors
Ds = [lo.diff(naive.shape, axis=axis) for axis in xrange(naive.ndim)]
# inversion
hypers = [1e6, 1e6, ]
y = tod.flatten()
M = P.T * W * P + np.sum([h * D.T * D for h, D in zip(hypers, Ds)])
x, conv = spl.cgs(M, P.T * W * y, callback=lo.CallbackFactory(verbose=True))
sol = naive.zeros(naive.shape)
sol[:] = x.reshape(sol.shape)
