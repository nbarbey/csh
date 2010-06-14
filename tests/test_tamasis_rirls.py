#!/usr/bin/env python
from tamasis import *
import numpy as np
import lo
import lo.iterative as it
import lo.pywt_lo as pywt_lo

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
# mask
M = lo.mask(backmap < 0).T
#M = lo.identity(2 * (backmap.size,))
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator()) * M
# prior
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64) * M
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64) * M
Dw = pywt_lo.wavelet2(backmap.shape, "haar") * M
# inversion
y = tod.flatten()
callback = lo.CallbackFactory(verbose=True)
x = it.rirls(P, (Dx ,Dy, Dw), y, p=1.5, maxiter=100, callback=callback)
sol = backmap.zeros(backmap.shape)
sol[:] = (M * x).reshape(sol.shape)
