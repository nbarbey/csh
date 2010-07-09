#!/usr/bin/env python
import numpy as np
import os
import tamasis as tm
import lo
import scipy.sparse.linalg as spl

datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]']
pacs = tm.PacsObservation(filename=filenames, 
                       fine_sampling_factor=1, keep_bad_detectors=False)
# reset pacs header to have a shape multiple of 4
#header = pacs.get_map_header()
#header['NAXIS1'] = 192
#header['NAXIS2'] = 192
#header['CRPIX1'] = 96
#header['CRPIX2'] = 96
# data
tod = pacs.get_tod()
# remove bad pixels (by updating mask !)
#tod = remove_bad_pixels(tod)
# deglitching
#projection = tm.Projection(pacs, header=header, resolution=3., npixels_per_sample=6)
projection = tm.Projection(pacs, resolution=3., npixels_per_sample=6)
tm.deglitch_l2mad(tod, projection)
# model
masking = tm.Masking(tod.mask)
model = masking * projection
# remove drift
tod = tm.filter_median(tod, length=999)

model = masking * projection
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
y = (masking.T * tod).flatten()
# algos
algos = [spl.cg, spl.cgs, spl.bicg, spl.bicgstab]
models = [P.T * P, P.T * P + Dx.T * Dx + Dy.T * Dy,]
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
                # if convergence not achieve output a nan
                n_iterations.append(np.nan)
            resid.append(callback.resid[-1])

n_iterations = np.asarray(n_iterations).reshape((len(models), len(algos))).T
resid = np.asarray(resid).reshape((len(models), len(algos))).T
