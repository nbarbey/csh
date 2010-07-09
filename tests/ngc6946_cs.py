#!/usr/bin/env python
import numpy as np
import os
import tamasis as tm
import lo
import csh


# ugly function ...
def or_int8(a, b):
    return np.clip(a % 2 + b % 2, 0, 1)

def remove_bad_pixels(tod):
    # hand remove bad pixels
    bp = np.array([22, 40, 45, 46, 81, 86, 134])
    X, Y = np.meshgrid(np.arange(tod.shape[1]), np.arange(tod.shape[0]))
    if tod.mask is None:
        tod.mask = np.zeros(tod.shape, dtype="int8")
    for i in bp:
        tod.mask = or_int8(tod.mask, (Y == i).astype('int8'))
    return tod

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]']
pacs = tm.PacsObservation(filename=filenames, 
                       fine_sampling_factor=1, keep_bad_detectors=False)
# reset pacs header to have a shape multiple of 4
header = pacs.get_map_header()
resolution = 3.
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
C = csh.cs(tod.shape, factor=8)
ctod = (C * tod.flatten()).reshape((tod.shape[0], tod.shape[1] / factor))
ctod = tm.Tod(ctod)
ctod.nsamples = [ns / factor for ns in tod.nsamples]
# uncompress for deglitching
uctod = tod.copy(tod.shape)
y0, t = lo.spl.cgs(C.T * C, C.T * ctod.flatten())
uctod[:] = y0.reshape(tod.shape)
#uctod[:] = (C.T * ctod.flatten()).reshape(tod.shape)

# deglitching
projection = tm.Projection(pacs, header=header, resolution=3., npixels_per_sample=6)
uctod.mask = tm.deglitch_l2mad(uctod, projection)
#ctod = compression.direct(uctod)
# model
masking = tm.Masking(uctod.mask)
model = masking * projection
# remove drift
#ctod = tm.filter_median(ctod, length=3000 / 8.)
# first map
M = C * lo.aslinearoperator(model.aslinearoperator())
#P = lo.aslinearoperator(projection.aslinearoperator())
#C = csh.averaging(tod.shape, factor=8)
#I = lo.mask(uctod.mask)
#M = C * I.T * I * P
#M = C * P
backmap = (M.T * ctod.flatten()).reshape(projection.shapein)
#weights = (M.T * np.ones(ctod.size)).reshape(projection.shapein)
weights = projection.transpose(tod.ones(tod.shape))
MM = lo.mask(weights == 0)
M = M * MM.T
# define algo
# priors
Dx = lo.diff(backmap.shape, axis=0, dtype=np.float64) * MM.T
Dy = lo.diff(backmap.shape, axis=1, dtype=np.float64) * MM.T
#Dw = lo.pywt_lo.wavedec2(backmap.shape, "haar", level=3)
# inversion
x, conv = lo.rls(M, (Dx, Dy), (1e0, 1e0),  ctod.flatten())
sol = tm.Map(np.zeros(backmap.shape))
sol[:] = (MM.T * x).reshape(sol.shape)
sol.header = header
# save
sol.writefits(os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',
                           'ngc6946_cs_rls.fits'))
