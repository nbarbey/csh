#!/usr/bin/env python
"""
This script show how one can build on TAMASIS and use home-made
estimation tools.
"""
# Imports
# --------
import os # to handle files
import numpy as np # numerical computations
from copy import copy # copy of objects
from tamasis import * # import everything from tamasis
import lo # my Linear Operators

# loading data
# -------------
filename = os.path.join(os.getenv('HOME'), 'data', 'pacs', 'frames_blue.fits')
# define a PacsObservation from data fits file
# this is level 1 data in HIPE format
pacs = PacsObservation(filename=filename, policy_bad_detector="keep")
# this method actually loads the data in memory as a Tod
# which is an ndarray subclass
tod = pacs.get_tod()

# Generating acquisition model
# -----------------------------
# projector
projection = Projection(pacs, # the PacsObseravtion object
                        resolution=3.2, # resolution of the sky map in arcsec
                        oversampling=False)
# backprojection
backmap = projection.T(tod)
# coverage map
coverage = projection.T(np.ones(tod.shape))
# naive map
naive = backmap / coverage
# mask according to coverage (everything that is covered by less than 10.)
mask = Masking(coverage < 10.)
# The model is the masking of the sky map then the projection
# This is basically matrix multiplication
model = projection * mask

# Performing inversion
# ---------------------

#  with TAMASIS
x_tm = mapper_rls(tod, model, hyper=1e-1, tol=1e-10, maxiter=100)

#  with lo routines
# transform to lo
H = lo.aslinearoperator(model * mask)
# smoothness priors
Ds = [lo.diff(backmap.shape, axis=axis) for axis in (0, 1)]
# inversion
y = tod.ravel() # requires 1d input
x_lo = lo.acg(H, y, Ds, 1e-1 * np.ones(3), tol=1e-10, maxiter=100)
x_lo.resize(backmap.shape) # output is 1d so need reshaping

# with sparsity assumptions (using Huber algorithm)
x_h = lo.hacg(H, y, Ds, 1e1 * np.ones(3),
            np.asarray((None, 1e-6, 1e-6, 1e-6)),
            x0=x_lo.flatten(), tol=1e-7, maxiter=200)
x_h.resize(backmap.shape)
