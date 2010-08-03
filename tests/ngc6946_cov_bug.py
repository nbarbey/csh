import os
import numpy as np
import lo
import tamasis as tm
import scipy.sparse.linalg as spl

def compress(tod, C, factor):
    ctod = (C * tod.flatten())
    ctod = ctod.reshape((tod.shape[0] ,tod.shape[1] / factor))
    ctod = tm.Tod(ctod)
    ctod.nsamples = [ns / factor for ns in tod.nsamples]
    return ctod

def uncompress(ctod, C, factor):
    uctod = tm.Tod(np.zeros((ctod.shape[0], ctod.shape[1] * factor)))
    y0, t = spl.cgs(C.T * C, C.T * ctod.flatten())
    uctod[:] = y0.reshape(uctod.shape)
    return uctod

def noise_covariance(tod, obs):
    """
    Defines the noise covariance matrix
    """
    length = 2 ** np.ceil(np.log2(np.array(tod.nsamples) + 200))
    invNtt = tm.InvNtt(length, obs.get_filter_uncorrelated())
    fft = tm.Fft(length)
    padding = tm.Padding(left=invNtt.ncorrelations,
                         right=length - tod.nsamples - invNtt.ncorrelations)
    masking = tm.Masking(tod.mask)
    cov = masking * padding.T * fft.T * invNtt * fft * padding * masking
#    cov = padding.T * fft.T * invNtt * fft * padding 
    return cov

# define data set
datadir = os.getenv('CSH_DATA')
filename = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67614]',
            datadir + '/1342185455_blue_PreparedFrames.fits[5954:67615]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
#compressions = ["", "ca", "cs"]
compressions = ["ca"]
# median filter length
filter_length = 10000
hypers = (1e10, 1e10)
resolution = 3.
factor = 4
ext = ".fits"
pre = "ngc6946_madmap1_"
# find a map for each compression and save it
obs = tm.PacsObservation(filename=filename,
                         fine_sampling_factor=1,
                         detector_policy='remove')
tod = obs.get_tod()
header = obs.get_map_header()
header.update('CDELT1', resolution / 3600)
header.update('CDELT2', resolution / 3600)
npix = 5
good_npix = False
projection = tm.Projection(obs, header=header,
                           resolution=resolution,
                           oversampling=False,
                           npixels_per_sample=npix)
model = projection
#C = csh.averaging(tod.shape, factor=factor)
compression_shape = [tod.size / factor, tod.size, ]
C = tm.CompressionAverage(compression_factor=factor).aslinearoperator(compression_shape)
C = lo.aslinearoperator(C)
# compress data
ctod = compress(tod, C, factor)
# uncompress for preprocessing
uctod = uncompress(ctod, C, factor)
# XXX need mask ...
uctod.mask = tod.mask
# deglitching
uctod = tm.filter_median(uctod, length=10)
uctod.mask = tm.deglitch_l2mad(uctod, projection)
masking = tm.Masking(uctod.mask)
model = masking * projection
# filtering
ctod = tm.filter_median(ctod, length=filter_length / factor)
cov = noise_covariance(ctod, obs)
S = cov.aslinearoperator()
