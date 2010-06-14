#!/usr/bin/env python
import numpy as np
from copy import copy
from tamasis import *
import lo

# data
pacs = PacsObservation(filename=tamasis_dir+'tests/frames_blue.fits',
                       fine_sampling_factor=1, 
                       keep_bad_detectors=False)
tod = pacs.get_tod()
# projector
projection = Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
model = projection
# naive map
backmap = model.transpose(tod)
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator())
# priors
Ds = [lo.diff(backmap.shape, axis=axis) for axis in xrange(backmap.ndim)]
Ds.append(lo.pywt_lo.wavelet2(backmap.shape, "haar"))
# inversion
y = tod.flatten()
x0 = lo.iterative.acg(P, Ds, (1e1, 1e1, 1e1), y)
sol0 = backmap.zeros(backmap.shape)
sol0[:] = x0.reshape(sol0.shape)

x = lo.iterative.hacg(P, Ds, (1e3, 1e3, 1e3), (1e6, 1e-6, 1e-6, 1e-6), y, x0=copy(x0))
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
