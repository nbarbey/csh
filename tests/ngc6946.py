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
# remove drift
tod = filter_median(tod, length=999)
# first map
backmap = model.transpose(tod)
P = lo.ndsubclass(backmap, tod, matvec=model.direct, rmatvec=model.transpose)
y = tod.flatten()
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar", level=3)
# inversion
x, conv = lo.rls(P, (Dx, Dy), (1e1, 1e1),  y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
# save
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'ngc6946_rls.fits'))
