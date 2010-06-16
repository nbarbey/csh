#!/usr/bin/env python
import tamasis as tm
import csh
import numpy as np
import lo
import scipy.sparse.linalg as spl

# data
pacs = tm.PacsObservation(filename=tm.tamasis_dir+'tests/frames_blue.fits',
                          fine_sampling_factor=1, keep_bad_detectors=True)
tod = pacs.get_tod()
# compression model
#C = lo.binning(tod.shape, factor=8, axis=1, dtype=np.float64)
shape = (64, 32) + (tod.shape[1], )
C = csh.binning3d( shape, factors=(2, 2, 2))
# compress data
ctod = C * tod.flatten()
# projector
projection = tm.Projection(pacs, resolution=3.2, oversampling=False,
                           npixels_per_sample=6)
model = projection
# naive map
backmap = model.transpose(tod)
# transform to lo
#P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
P = lo.aslinearoperator(model.aslinearoperator())
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

# L2 score
bin_score = csh.score(A.T * A)
print("score of binning3d strategy " + str(bin_score))
