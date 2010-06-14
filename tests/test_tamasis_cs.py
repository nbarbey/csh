#!/usr/bin/env python
from tamasis import *
from csh import *
import numpy as np
import lo
import scipy.sparse.linalg as spl

# data
pacs = PacsObservation(filename=tamasis_dir+'tests/frames_blue.fits',
                       fine_sampling_factor=1,
                       keep_bad_detectors=True,
                       mask_bad_line=True
                       )
tod = pacs.get_tod()
# set bad detectors to 0
bd_mask = pacs.bad_detector_mask.flatten()
for i in xrange(tod.shape[1]): tod[bd_mask, i] = 0
# compression model
C = cs(tod.shape, 8)
# compress data
ctod = C * tod.flatten()
# projector
projection = Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
model = projection
# naive map
backmap = model.transpose(tod)
# transform to lo
P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
# full model
A = C * P
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar")
# inversion
y = ctod.flatten()
x, conv = lo.rls(A, (Dx, Dy), (1e1, 1e1),  y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
