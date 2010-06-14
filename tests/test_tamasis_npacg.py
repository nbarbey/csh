#!/usr/bin/env python
from tamasis import *
import numpy as np
import lo
import scipy.sparse.linalg as spl

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
P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
# priors
Dx = lo.diff(backmap.shape, axis=0)
Dy = lo.diff(backmap.shape, axis=1)
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar")
# inversion
y = tod.flatten()
x = lo.iterative.npacg(P, (Dx, Dy), (1e1, 1e1), (2, 1.5, 1.5), y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
