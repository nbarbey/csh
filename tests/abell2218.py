#!/usr/bin/env python
import os
import pyfits
from tamasis import *
import lo
from csh import *
# define data set
datadir = os.getenv('CSH_DATA')
ids = ['1342184518', '1342184519', '1342184596', '1342184597', 
       '1342184598', '1342184599']
filenames = [os.path.join(datadir, id_str + '_blue_PreparedFrames.fits')
             for id_str in ids]
pacs = PacsObservation(filename=filenames, 
                       fine_sampling_factor=1, keep_bad_detectors=False)
# get header from altieri maps
header = pyfits.fitsopen('/mnt/herschel1/mapmaking/nbarbey/Abell2218_altieri/' +
                         'a2218_red_Map.v2.2.sci.fits')[0].header
# data
tod = pacs.get_tod()
# remove bad pixels (by updating mask !)
#tod = remove_bad_pixels(tod)
# deglitching
projection = Projection(pacs, header=header, resolution=3., npixels_per_sample=6)
tod.mask = deglitch_l2mad(tod, projection)
# model
masking = Masking(tod.mask)
model = masking * projection
# remove drift
tod = filter_median(tod, length=49)
# first map
backmap = model.transpose(tod)
P = lo.aslinearoperator(model.aslinearoperator())
y = tod.flatten()
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64)
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64)
# inversion
x, conv = lo.rls(P, (Dx, Dy), (1e2, 1e2),  y)
sol = backmap.zeros(backmap.shape)
sol[:] = x.reshape(sol.shape)
# save
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'abell2218.fits'))
