#!/usr/bin/env python
import numpy as np
import tamasis as tm
import lo
import csh.filter as filt

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
N = filt.noise_filter(tod.shape, kernel)
# apply to data
yn = N.T * tod.flatten()
# apply to model
M = P * N
# priors
Ds = [lo.diff(backmap.shape, axis=axis) for axis in xrange(backmap.ndim)]
Ds.append(lo.pywt_lo.wavelet2(backmap.shape, "haar"))
# inversion
y = tod.flatten()
x, conv = lo.rls(M, Ds, (1e1, 1e1, 1e-1),  y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
