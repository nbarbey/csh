#!/usr/bin/env python
import numpy as np
import tamasis as tm
import lo
import scipy.sparse.linalg as spl

# data
pacs = tm.PacsObservation(filename=tm.tamasis_dir+'tests/frames_blue.fits')
tod = pacs.get_tod()
# projector
projector = tm.Projection(pacs, resolution=3.2, oversampling=False, npixels_per_sample=6)
masking_tod  = tm.Masking(tod.mask)
model = masking_tod * projector
# naive map
backmap = model.transpose(tod)
# coverage map
weights = model.transpose(tod.ones(tod.shape))
# mask on map
mask = weights == 0
M = lo.mask(mask)
# preconditionner
iweights = 1 / weights
iweights[np.where(np.isfinite(iweights) == 0)] = 0.
M0 = lo.diag(iweights.flatten())
# transform to lo
P = lo.aslinearoperator(model.aslinearoperator())
# priors
Dx = lo.diff(backmap.shape, axis=0)
Dy = lo.diff(backmap.shape, axis=1)
# inversion
y = (masking_tod.T * tod).flatten()
# algos
algos = [spl.cg, spl.cgs, spl.bicg, spl.bicgstab]
models = [P.T * P, 
          P.T * P + Dx.T * Dx + Dy.T * Dy,
          M * P.T * P * M.T,
          M * (P.T * P  + Dx.T * Dx + Dy.T * Dy) * M.T,
          ]
n_iterations = []
resid = []
for algo in algos:
    for A in models:
        callback = lo.CallbackFactory(verbose=True)
        is_masked = A.shape[0] == 5136
        if is_masked:
            b = M * P.T * y
            M1 = M * M0 * M.T
        else:
            b = P.T * y
            M1 = M0
        x, conv = algo(A, b, M=M1, maxiter=1000, callback=callback)
        if conv == 0:
            n_iterations.append(callback.iter_[-1])
        else:
            # if convergence not achieve output a nan
            n_iterations.append(np.nan)
        resid.append(callback.resid[-1])

n_iterations = np.asarray(n_iterations).reshape((len(models), len(algos)))
resid = np.asarray(resid).reshape((len(models), len(algos)))
