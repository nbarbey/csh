#!/usr/bin/env python
from tamasis import *
import numpy as np
import interface as LO
import iterative
import scipy.sparse.linalg as spl
import pywt_lo

# data
pacs = PacsObservation(filename=tamasis_dir+'tests/frames_blue.fits',
                       fine_sampling_factor=1, 
                       keep_bad_detectors=False)
tod = pacs.get_tod()
# projector
#projection = Projection(pacs, resolution=3.2, finer_sampling=False, npixels_per_sample=6)
projection = Projection(pacs, resolution=1., finer_sampling=False, npixels_per_sample=17)
model = projection
# naive map
backmap = model.transpose(tod)
# mask
unity = Tod.ones(tod.shape, nsamples=tod.nsamples)
weights = model.transpose(unity)
map_naive = backmap / weights
# transform to LO
P = LO.ndsubclass(backmap, tod, model.direct, model.transpose)
# prior
Dx = LO.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = LO.diff(backmap.shape, axis=1, dtype=np.float64)
#Dw = pywt_lo.wavedec2(backmap.shape, "haar")
# inversion
y = P.T * tod.flatten()
M = P.T * P + 1e1 * (Dx.T * Dx + Dy.T * Dy)  #+ 1e0 * (Dw.T * Dw)
callback = iterative.Callback(M, y)
x, conv = spl.bicgstab(M, y, callback=callback)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
