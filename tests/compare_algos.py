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
models = [P.T * P,  P.T * P + Dx.T * Dx + Dy.T * Dy]
n_iterations = []
resid = []
for algo in algos:
    for A in models:
        for is_masked in (False, True):
            callback = lo.CallbackFactory(verbose=True)
            if is_masked:
                A = M * A * M.T
                b = M * P.T * y
                M1 = M * M0 * M.T
            else:
                b = P.T * y
                M1 = M0
            x, conv = algo(A, b, M=M1, maxiter=1000, callback=callback)
            if conv == 0:
                n_iterations.append(callback.iter_[-1])
            else:
                # if convergence not achieved output a nan
                n_iterations.append(np.nan)
            resid.append(callback.resid[-1])

n_iterations = np.asarray(n_iterations).reshape((2 * len(models), len(algos))).T
resid = np.asarray(resid).reshape((2 * len(models), len(algos))).T
