#!/usr/bin/env python

"""
Test mini-mapmaking matrix on simulated data
"""
import os
import numpy as np
import tamasis as tm
import csh # my PACS inversion routines
import fitsarray as fa
import lo

# scan parameters
ra0, dec0 = 30., 30.
pointing_params = {
    'cam_angle':0.,
    "scan_angle":0.,
    "scan_nlegs":1,
    "scan_length":400.,
    "scan_speed":60.0,
    "compression_factor":1 # we will compress later
    }
pointing = tm.pacs_create_scan(ra0, dec0, **pointing_params)
# create obs object
obs = tm.PacsSimulation(pointing, "red", policy_bad_detector="keep")
# we dont need 200 first and last frames
obs.pointing.removed[200:] = True
obs.pointing.removed[-201:] = True
# create projector
projection = tm.Projection(obs, npixels_per_sample=4)
P = lo.aslinearoperator(projection)
# simulate data
x0 = tm.gaussian(projection.shapein, 3) # map with gaussian source
tod0 = projection(x0) # data
n = np.random.randn(*tod0.shape) # noise
nsr = 1e-2
tod = tod0 + nsr * n # noisy data
y = tod.ravel() # as 1d array

# load compression matrix
filename = os.path.join(os.getenv("HOME"), "data", "pacs",
                        "mmc_cam_angle_0_scan_angle_0_speed60.fits")
c = fa.FitsArray(file=filename).astype(np.float64)
cmm = csh.compression.AnyOperator(c)
C = cmm((projection.shapeout[0], projection.shapeout[1][0]),)
# compress
z = C * y

# inversion
H = C * P
Ds = [lo.diff(x0.shape, axis=i) for i in (0, 1)]
x_inv = lo.acg(H, z, Ds, 1e-1 *np.ones(2), tol=1e-10, maxiter=100)
x_inv.resize(x0.shape)

# condition number
#M = H.T * H
#Md = M.todense()
#print np.linalg.cond(Md)
#print lo.iterative.utils.cond(H.T * H)
