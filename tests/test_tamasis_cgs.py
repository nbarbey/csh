#!/usr/bin/env python
import numpy as np
import scipy.sparse.linalg as spl
import tamasis as tm
import lo

# data
pacs = tm.PacsObservation(filename=tm.tamasis_dir+'tests/frames_blue.fits')
tod = pacs.get_tod()
# projector
#model = tm.Projection(pacs, resolution=3.2, finer_sampling=False, npixels_per_sample=6)
model = tm.Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
# naive map
backmap = model.transpose(tod)
# transform to lo
#P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
P = lo.aslinearoperator(model.aslinearoperator())
# priors
Ds = [lo.diff(backmap.shape, axis=axis) for axis in xrange(backmap.ndim)]
Ds.append(lo.pywt_lo.wavelet2(backmap.shape, "haar"))
# inversion
hypers = [1e1, 1e1, 1e-1]
y = tod.flatten()
M = P.T * P + np.sum([h * D.T * D for h, D in zip(hypers, Ds)])
x, conv = spl.cgs(M, P.T * y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
