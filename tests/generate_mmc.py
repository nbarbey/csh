#!/usr/bin/env python

"""
Generate a compression matrix using a local mapmaking operator.
"""
# Imports are always first traditionally, to show dependencies :
import os # to handle files / directories
import numpy as np # low-level numerical package (matrices, fft, ...)
import tamasis as tm # TAMASIS package
import lo # My linear operators and optimization routines
# to install :
# git clone git://github.com/nbarbey/lo.git
# cd lo
# python setup.py install
# # or
# python setup.py install --prefix=$HOME/local/ # (for exemple)
import fitsarray as fa # to easily handle FITS files (depends on pyfits)
# to install :
# git clone git://github.com/nbarbey/fitsarray.git
# cd fitsarray
# python setup.py install

# define projection matrix
# ------------------------
# center of the map
ra0, dec0 = 30., 30.
# let us define a small scan with 1 leg for testing purposes :
pointing_params = {
    'cam_angle':0.,
    "scan_angle":0.,
    "scan_nlegs":1,
    "scan_length":4.,
    "compression_factor":1,
    "scan_speed":60.0
    }
pointing = tm.pacs_create_scan(ra0, dec0, **pointing_params)
# create obs object
obs = tm.PacsSimulation(pointing, "red", policy_bad_detector="keep")
# to define my compression matrix I need only 8 frames
# so I mask the others :
obs.pointing.removed[:] = True
obs.pointing.removed[201:209] = False
# create projector
projection = tm.Projection(obs)
P = lo.aslinearoperator(projection)

# define masking of coefficients
# ------------------------------
# number of coefficients to keep
factor = 8.
nk = P.shape[0] / factor
# coverage map
w = P.T * np.ones(P.shape[0])
# sort coverage map coefficients
ws = np.sort(w)
# find the nk largest coverage elements
threshold = ws[-nk]
mask = w < threshold
# define decimation Linear Operator according to coverage map coef
# selection.
M = lo.decimate(mask)
# model is projection matrix time decimation matrix
H = P * M.T

# Store model matrix on hard-drive as FITS
# ---------------------------------------------
# convert LinearOperator to dense matrix (ndarray)
Hd = H.todense()
Hd = np.asmatrix(Hd)
# convert into my FitsArray :
Hd_fits = fa.FitsArray(data=Hd)
# save the model to defined filename
filename = os.path.join(os.getenv("HOME"), "data", "pacs",
                        "mmc_model_cam_angle_0_scan_angle_0_speed60.fits")
Hd_fits.tofits(filename)

# Define and store mini ma-pmaking matrix
# ---------------------------------------
# (H^T H + a D^T D)^{-1} H^T
# prior is "smoothness" along each axis (0 and 1)
D = [lo.diff(projection.shapein, axis=i) for i in (0, 1)]
# apply the same decimation to priors
D = [Di * M.T for Di in D]
# can sum LinearOperators
DD = sum([Di.T * Di for Di in D])
# convert to dense matrix (ndarray)
DDd = DD.todense()
# Use numpy routines to compute the exact dense map-making matrix.
# This is possible since we have only 8 frames here.
# Otherwise conjugate gradient inversions are mandatory.
# The following line can be longer (approx 1 minute).
H_inv = np.linalg.inv(Hd.T * Hd + DDd) * Hd.T
# save to FITS files
H_inv_fits = fa.FitsArray(data=H_inv)
filename = os.path.join(os.getenv("HOME"), "data", "pacs",
                        "mmc_cam_angle_0_scan_angle_0_speed60.fits")
H_inv_fits.tofits(filename)
