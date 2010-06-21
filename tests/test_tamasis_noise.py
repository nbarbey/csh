#!/usr/bin/env python
import numpy as np
import tamasis as tm
import lo
import csh.filter as filt
from time import time
import scipy.sparse.linalg as spl

# data
pacs = tm.PacsObservation(filename=tm.tamasis_dir+'tests/frames_blue.fits')
tod = pacs.get_tod()
# projector
model = tm.Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
# naive map
backmap = model.transpose(tod)
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator())
# derive filter
kernel = filt.kernel_from_tod(tod, length=10)
#kern = np.mean(kernel, axis=0)
N = filt.kernels_convolve(tod.shape, 1 / np.sqrt(kernel))
# apply to data
yn = N * tod.flatten()
# apply to model
M = N * P
# priors
Ds = [lo.diff(backmap.shape, axis=axis) for axis in xrange(backmap.ndim)]
#Ds.append(lo.pywt_lo.wavelet2(backmap.shape, "haar"))
# inversion
#y = tod.flatten()
x, conv = lo.rls(M, Ds, (1e1, 1e1, 1e-1),  yn, spl.bicgstab)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
