#!/usr/bin/env python
from tamasis import *
from csh import *
import numpy as np
import lo
import scipy.sparse.linalg as spl

# data
pacs = PacsObservation(filename=tamasis_dir+'tests/frames_blue.fits',
                       fine_sampling_factor=1,
                       keep_bad_detectors=False)
tod = pacs.get_tod()
# compression model
C = lo.binning(tod.shape, factor=8, axis=1, dtype=np.float64)
# compress data
ctod = C * tod.flatten()
# projector
projection = Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
model = projection
# naive map
backmap = model.transpose(tod)
# coverage map
weights = model.transpose(tod.ones(tod.shape))
# mask on map
mask = weights == 0
M = lo.mask(mask)
# transform to lo
#P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
P = lo.aslinearoperator(model.aslinearoperator())
# full model
A = C * P * M.T
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64) * M.T
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64) * M.T
Dw = lo.pywt_lo.wavelet2(backmap.shape, "haar") * M.T
# inversion
y = ctod.flatten()
x, conv = lo.rls(A, (Dx, Dy, Dw), (1e1, 1e1, 1e1),  y)
sol = backmap.zeros(backmap.shape)
sol[mask == 0] = x
