#!/usr/bin/env python
import os
from tamasis import *
import lo
from csh import *
import csh.filter as filt
from time import time
import scipy.sparse.linalg as spl

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]'
             ]
pacs = PacsObservation(filename=filenames, 
                       fine_sampling_factor=1, keep_bad_detectors=False)
# reset pacs header to have a shape multiple of 4
header = pacs.get_map_header()
header['NAXIS1'] = 192
header['NAXIS2'] = 192
header['CRPIX1'] = 96
header['CRPIX2'] = 96
# data
tod = pacs.get_tod()
# remove bad pixels (by updating mask !)
#tod = remove_bad_pixels(tod)
# deglitching
projection = Projection(pacs, header=header, resolution=3., npixels_per_sample=6)
deglitch_l2mad(tod, projection)
# model
masking = Masking(tod.mask)
model = masking * projection
P = lo.aslinearoperator(model.aslinearoperator())
# derive filter
#tod = filter_median(tod, length=9999)
#kernel = filt.kernel_from_tod(tod, length=1000)
kernel =  (1 + (10. / np.arange(500)) ** .25)
kernel = np.concatenate((kernel[::-1], kernel))
#kern = np.mean(kernel, axis=0)
N = filt.kernels_convolve(tod.shape, 1 / np.sqrt(kernel))
# apply to data
yn = N * tod.flatten()
# apply to model
M = N * P
# first map
backmap = model.transpose(tod)
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar", level=3)
# inversion
x, conv = lo.rls(M, (Dx, Dy), (1e1, 1e1),  yn, tol=1e-10)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
# save
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'ngc6946__filter_rls.fits'))
