#!/usr/bin/env python
from tamasis import *
import numpy as np
import lo
import lo.iterative_thresholding as it
import lo.pywt_lo as pywt_lo

# data
pacs = PacsObservation(filename=tamasis_dir+'tests/frames_blue.fits',
                       fine_sampling_factor=1, 
                       keep_bad_detectors=False)
tod = pacs.get_tod()
# projector
header = pacs.get_map_header()
header['NAXIS1'] = 96
header['NAXIS2'] = 96
projection = Projection(pacs, header=header, resolution=3.2, oversampling=False, npixels_per_sample=6)
model = projection
# naive map
backmap = model.transpose(tod)
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator())
# prior
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
Dw = pywt_lo.wavedec2(backmap.shape, "haar", level=5)
# inversion
y = tod.flatten()
x = it.fista(P, Dw, y, mu=1e-4, nu=1e-2, maxiter=1000)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
