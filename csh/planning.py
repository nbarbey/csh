"""
To perform planning of measurements.
"""
import numpy as np
import tamasis as tm
import lo

def generate_model(ra0, dec0, pointing_params, repeats=1, cross_scan=False,
                       span_angles=False, band="red", map_header=None):
    """
    Generate a PACS projector.
    """
    pointing1 = tm.pacs_create_scan(ra0, dec0, **pointing_params)
    # create obs object
    obs1 = tm.PacsSimulation(pointing1, band)
    if map_header is None:
        map_header = obs1.get_map_header()        
    # create projector
    projection1 = tm.Projection(obs1, header=map_header)
    P = lo.aslinearoperator(projection1)
    # cross scan
    if cross_scan:
        pointing_params["scan_angle"] += 90.
        pointing2 = tm.pacs_create_scan(ra0, dec0, **pointing_params)
        obs2 = tm.PacsSimulation(pointing2, band)
        projection2 = tm.Projection(obs2, header=map_header)
        P2 = lo.aslinearoperator(projection2)
        P = lo.concatenate((P, P2))
    # repeats
    if repeats > 1:
        P = lo.concatenate((P, ) * repeats)
    if span_angles:
        raise NotImplemented
    # noise
    N = generate_noise_filter(obs1, P, band)

    # covered area
    map_shape = map_header['NAXIS2'], map_header['NAXIS1']
    coverage = (P.T * np.ones(P.shape[0])).reshape(map_shape)
    seen = coverage != 0
    M = lo.decimate(coverage < 10)
    # global model
    H = N * P * M.T
    return H

def generate_noise_filter(obs1, P, band):
    # data
    if band == "red":
        shape0 = 512.
    if band == "green" or band == "blue":
        shape0 = 2048.
    nsamples = P.shape[0] / shape0
    # 1/f noise model
    length = 2 ** np.ceil(np.log2(np.array(nsamples)))
    # square root in Fourier !!!
    filt = obs1.get_filter_uncorrelated()
    fft0 = lo.fft2(filt.shape, axes=(1,))
    filt2 = np.real(fft0.T(np.sqrt(fft0(filt))))
    invNtt = tm.InvNtt(length, filt2)
    fft = tm.FftHalfComplex(length)
    padding = tm.Padding(left=invNtt.ncorrelations,
                         right=length - nsamples - invNtt.ncorrelations)
    cov = padding.T * fft.T * invNtt * fft * padding
    N12 = lo.aslinearoperator(cov)
    return N12

def cond(H, dense=False, **kwargs):
    if dense:
        return np.linalg.cond((H.T * H).todense())
    else:
        from lo.iterative import utils
        Ha = utils.eigendecomposition(H, which="BE", k=2, **kwargs)
        return Ha.cond()
