#!/usr/bin/env python
import os
from tamasis import *
import lo
from csh import *
# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]']
pacs = PacsObservation(filename=filenames, 
                       fine_sampling_factor=1, keep_bad_detectors=False)
# reset pacs header to have a shape multiple of 4
resolution = 3.
header = pacs.get_map_header()
#header['NAXIS1'] = 192
#header['NAXIS2'] = 192
#header['CRPIX1'] = 96
#header['CRPIX2'] = 96
header.update('CDELT1', resolution / 3600)
header.update('CDELT2', resolution / 3600)
# data
tod = pacs.get_tod()
# remove bad pixels (by updating mask !)
#tod = remove_bad_pixels(tod)
# deglitching
projection = Projection(pacs, header=header, resolution=resolution,
                        npixels_per_sample=5)
deglitch_l2mad(tod, projection)
# model
masking = Masking(tod.mask)
model = masking * projection
# remove drift
#tod = filter_median(tod, length=3000)
# first map
backmap = model.transpose(tod)
# coverage
weights = model.transpose(tod.ones(tod.shape))
P = lo.aslinearoperator(model.aslinearoperator())
# mask not seen part of the map
MM = lo.mask(weights ==0)
y = tod.flatten()
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar", level=3)
# inversion
x, conv = lo.rls(P * MM.T, (Dx * MM.T, Dy * MM.T), (1e1, 1e1),  y)
sol = backmap.zeros(backmap.shape)
sol[:] = (MM.T * x).reshape(sol.shape)
# save
sol.header = header
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'ngc6946_rls.fits'))
weights.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                               'ngc6946_rls_weights.fits'))
