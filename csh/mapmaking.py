"""A module that implements map-making methods
"""
import numpy as np
import tamasis as tm
import lo
import csh

def rls(filename, compression=None, factor=4, hypers=(1., 1.),
        deltas=(1e-8, 1e-8), header=None, wavelet=None,
        covariance=True, map_mask=True, deglitch=True, bpj=False,
        deglitch_filter_length=10, filtering=True,
        filter_length=10000, decompression=True, model_only=False,
        algo=lo.rls, optimizer=lo.spl.bicgstab, maxiter=None, tol=1e-5):
    """ Performs regularized least square map making
    """
    # load data and projection matrix
    tod, projection, header, obs = load_data(filename, header=header)
    model = projection
    # define compression
    if compression is None or compression == "" or compression == "no":
        C = lo.identity(2 * (tod.size, ))
        factor = 1
    elif compression == "ca":
        C = csh.averaging(tod.shape, factor=factor)
    elif compression == "cs":
        C = csh.cs(tod.shape, factor=factor)
    elif compression == "dt":
        C = csh.decimate_temporal(tod.shape, factor)
    # compress data
    ctod = compress(tod, C, factor)
    # deglitching
    if deglitch is True:
        # uncompress for preprocessing
        uctod = uncompress(ctod, C, factor)
        # XXX need mask ...
        uctod.mask = tod.mask
        uctod = tm.filter_median(uctod, length=deglitch_filter_length)
        #uctod_mask = tm.deglitch_l2mad(uctod, projection)
        #uctod = uncompress(ctod, C, factor)
        #uctod.mask = uctod_mask
        masking = tm.Masking(uctod.mask)
        model = masking * projection
        # remove glitches by interpolation for the covariance
        #uctod = tm.interpolate_linear(uctod)
        #uctod = uctod * uctod.mask
        # compress back the data
        #ctod = compress(uctod, C, factor)
    # median filtering
    if filtering is True:
        ctod = tm.filter_median(ctod, length=filter_length / factor)
    # model with compression
    M = lo.aslinearoperator(model.aslinearoperator())
    if decompression is True:
        M = C * M
    else:
        # ad-hoc compression model (assumes wrongly no compression)
        C0 = csh.decimate_temporal(tod.shape, factor=factor)
        M = C0 * M
    # noise covariance
    if covariance:
        cov = noise_covariance(ctod, obs)
        S = lo.aslinearoperator(cov.aslinearoperator())
        #S = lo.identity(2 * (ctod.size,))
    else:
        S = None
    # backprojection
    backmap = tm.Map((M.T * ctod.flatten()).reshape(projection.shapein))
    # priors
    Ds = [lo.diff(backmap.shape, axis=0, dtype=np.float64),]
    Ds.append(lo.diff(backmap.shape, axis=1, dtype=np.float64))
    if wavelet is not None:
        Ds.append(lo.pywt_lo.wavelet2(backmap.shape, wavelet, level=3, dtype=np.float64))
    # weights
    weights = projection.transpose(tod.ones(tod.shape))
    # masking the map
    if map_mask is True:
        MM = lo.mask(weights == 0)
        M = M * MM.T
        Ds = [D * MM.T for D in Ds]
    # to use the defined model for other purposes
    if model_only:
        return M
    # inversion
    #x, conv = algo(M, ctod.flatten(), Ds=Ds, hypers=hypers, deltas=deltas, S=S,
    #               optimizer=optimizer, maxiter=maxiter, tol=tol)
    x, conv = algo(M, ctod.flatten(), Ds=Ds, hypers=hypers, deltas=deltas,
                   maxiter=maxiter, tol=tol)
    # reshape map
    sol = tm.Map(np.zeros(backmap.shape))
    if map_mask:
        sol[:] = (MM.T * x).reshape(sol.shape)
    else:
        sol[:] = x.reshape(sol.shape)
    sol.header = header
    if bpj:
        backmap /= weights
        backmap[np.isnan(backmap)]= 0.
        return sol, backmap
    else:
        return sol

def compress(tod, C, factor):
    ctod = (C * tod.flatten())
    ctod = ctod.reshape((tod.shape[0] ,tod.shape[1] / factor))
    ctod = tm.Tod(ctod)
    ctod.nsamples = [ns / factor for ns in tod.nsamples]
    return ctod

def uncompress(ctod, C, factor):
    uctod = tm.Tod(np.zeros((ctod.shape[0], ctod.shape[1] * factor)))
    y0, t = lo.spl.cgs(C.T * C, C.T * ctod.flatten())
    uctod[:] = y0.reshape(uctod.shape)
    return uctod

def load_data(filename, header=None, resolution=3.):
    obs = tm.PacsObservation(filename=filename,
                              fine_sampling_factor=1,
                              detector_policy='remove')
    tod = obs.get_tod()
    if header is None:
        header = obs.get_map_header()
    header.update('CDELT1', resolution / 3600)
    header.update('CDELT2', resolution / 3600)
    npix = 5
    good_npix = False
    while good_npix is False:
        try:
            projection = tm.Projection(obs, header=header,
                                       resolution=resolution,
                                       oversampling=False,
                                       npixels_per_sample=npix)
            good_npix = True
        except(RuntimeError):
            npix +=1
    return tod, projection, header, obs

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
