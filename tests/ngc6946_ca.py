#!/usr/bin/env python
import numpy as np
import os
import tamasis as tm
import lo
import csh
# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]']
pacs = tm.PacsObservation(filename=filenames, 
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
y = tod.flatten()
# remove bad pixels (by updating mask !)
#tod = remove_bad_pixels(tod)
# compress data
factor = 8
compression = tm.CompressionAverage(factor)
C = lo.aslinearoperator(compression.aslinearoperator(shape=(tod.size / factor, tod.size)))
ctod = compression.direct(tod)
# uncompress for deglitching
uctod = tod.copy(tod.shape)
y0, t = lo.spl.cgs(C.T * C, C.T * ctod.flatten())
uctod[:] = y0.reshape(tod.shape)
# deglitching
projection = tm.Projection(pacs, header=header, resolution=3., npixels_per_sample=5)
uctod.mask = tm.deglitch_l2mad(uctod, projection)
ctod = compression.direct(uctod)
# model
masking = tm.Masking(uctod.mask)
model = compression * masking * projection
# remove drift
#ctod = tm.filter_median(ctod, length=3000 / 8.)
# first map
M = lo.aslinearoperator(model.aslinearoperator())
#P = lo.aslinearoperator(projection.aslinearoperator())
#C = csh.averaging(tod.shape, factor=8)
#I = lo.mask(uctod.mask)
#M = C * I.T * I * P
#M = C * P
backmap = model.transpose(ctod)
weights = model.transpose(ctod.ones(ctod.shape))
MM = lo.mask(weights ==0)
M = M * MM.T
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64) * MM.T
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64) * MM.T
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar", level=3)
# inversion
x, conv = lo.rls(M, (Dx, Dy), (1e0, 1e0),  ctod.flatten())
sol = backmap.zeros(backmap.shape)
sol[:] = (MM.T * x).reshape(sol.shape)
# save
sol.header = header
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'ngc6946_ca_rls.fits'))
