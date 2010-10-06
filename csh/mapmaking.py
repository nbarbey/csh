"""A module that implements map-making methods.
"""
import numpy as np
import tamasis as tm
import lo
import csh

# pipeline class
class Pipeline(object):
    """
    A class to define a list of processing steps (a pipeline).
    """
    def __init__(self, tasks, verbose=False):
        self.state = dict()
        self.tasks = tasks
        self.verbose = verbose

    def __call__(self, filename):
        import copy
        out = copy.copy(filename)
        for task in self.tasks:
            if self.verbose:
                print "Performing task " + str(task)
                #print self.state
            out = task(out, self.state)
        return out

    def __repr__(self):
        return repr(self.tasks)

class Task(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, data, state):
        return data

class LoadData(Task):
    def __init__(self, header=None, resolution=3.):
        self.header = header
        self.resolution = resolution

    def __call__(self, filename, state):
        # load data
        data, projection, header, obs = load_data(filename, 
                                                  header=self.header, 
                                                  resolution=self.resolution)
        # update state
        state['data'] = data
        state['model'] = projection
        state['header'] = header
        state['obs'] = obs
        state['resolution'] = self.resolution
        state['data_shape'] = data.shape
        state['mask'] = data.mask
        return data

class CompressData(Task):
    def __init__(self, compression, factor):
        self.compression = compression
        self.factor = factor

    def __call__(self, data, state):
        data = state['data']
        data_shape = state['data_shape']
        compression = self.compression
        if compression is None or compression == "" or compression == "no":
            C = lo.identity(2 * (data.size, ))
            factor = 1
        else:
            C = compression(data_shape, self.factor)
        # compress data
        data = compress(data, C, self.factor)
        # update pipeline state
        state['data'] = data
        state['compression'] = C
        state['compression_factor'] = self.factor
        state['compressed_data_shape'] = data.shape
        return data

class Deglitching(Task):
    def __init__(self, filter_length):
        self.filter_length = filter_length

    def __call__(self, data, state):
        # get parameters
        C = state.get('compression')
        factor = state.get('compression_factor', 1)
        # decompress
        uc_data = uncompress(data, C, factor)
        uc_data.mask = state['mask']
        # filter
        uc_data = tm.filter_median(uc_data, length=self.filter_length)
        tm.deglitch_l2mad(uc_data, state['model'])
        # define mask
        masking = tm.Masking(uc_data.mask)
        # update model
        state['model'] = masking * state['model']
        state['deglitching_filter_length'] = self.filter_length
        return data

class NoiseFiltering(Task):
    def __init__(self, filter_length, myfilter=tm.filter_median):
        self.filter_length = filter_length
        self.filter = myfilter

    def __call__(self, data, state):
        factor = state.get('factor', 1.)
        data = self.filter(data, length = self.filter_length)
        state['noise_filter_length'] = self.filter_length
        state['data'] = data
        return data

class Backprojection(Task):
    def __call__(self, data, state):
        M = state['model']
        bpj = M.T * data
        state['map'] = bpj
        return data

class DefineWeights(Task):
    def __call__(self, data, state):
        M = state['model']
        ones_data = np.ones(data.shape)
        weights = M.T * ones_data
        state['weights'] = weights
        return data

class WeightedBackprojection(Task):
    def __call__(self, data, state):
        self.bpj(data, state)
        self.weights(data, state)
        state['map'] /= state['weights']
        return data

class AsLinearOperator(Task):
    def __call__(self, data, state):
        mymap = state.get('map')
        model = state['model']
        M = lo.aslinearoperator(model.aslinearoperator())
        data = data.ravel()
        mymap = np.ravel(mymap)
        state['lo_data_shape'] = model.shapeout
        state['map_shape'] = model.shapein
        state['model'] = M
        state['data'] = data
        state['map'] = mymap
        return data

class CompressionModel(Task):
    def __init__(self, compression_model):
        self.model = compression_model

    def __call__(self, data, state):
        C = self.model(state['data_shape'],
                       factor=state['compression_factor'])
        state['compression_model'] = C
        state['model'] = C * state['model']
        return data

class NoiseModel(Task):
    def __call__(self, data, state):
        obs = state['obs']
        cov = noise_covariance(data, obs)
        S = lo.aslinearoperator(cov.aslinearoperator())
        state['noise_model'] = S
        return data

class MapMaskModel(Task):
    def __call__(self, data, state):
        if not 'weights' in state:
            get_weights = DefineWeights()
            get_weights(data, state)

        weights = state['weights']    
        MM = lo.mask(weights == 0)
        M = state['model']
        M = M * MM.T
        state['model'] = M
        if 'prior_models' in state:
            Ds = state['prior_models']
            Ds = [D * MM.T for D in Ds]
            state['prior_models'] = Ds
        state['map_mask_model'] = MM
        return data

class DefineSmoothnessPriors(Task):
    def __call__(self, data, state):
        map_shape = state['map_shape']
        ndim = len(map_shape)
        Ds = [lo.diff(map_shape, axis=i) for i in xrange(ndim)]
        state['prior_models'] = Ds
        return data

class Inversion(Task):
    def __init__(self, algo, **kwargs):
        self.algo = algo
        self.kwargs = kwargs

    def __call__(self, data, state):
        M = state['model']
        Ds = state.get('prior_models', [])
        x = self.algo(M,
                      np.asarray(data),
                      Ds=Ds,
                      **self.kwargs)
        # reshape map
        map_shape = state['map_shape']
        if 'map_mask_model' in state:
            MM = state['map_mask_model']
            sol = (MM.T * x).reshape(map_shape)
        else:
            sol = x.reshape(map_shape)
        header = state['header']
        state['map'] = tm.Map(sol, header=header)
        return data

class SaveMap(Task):
    """
    Save the current map into filename in fits format.

    Parameters
    ----------
    filename : string
       Name of the file where the map will be saved.

    Returns
    -------
    savemap: a Task instance which will save the map to filename.
    """
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, data, state):
        mymap = state['map']
        mymap.writefits(self.filename)
        return data

def get_pipeline(**kwargs):
    """
    Defines a pipeline from a set of parameters. If a parameter is
    present, it will add the corresponding task to the pipeline
    automatically.
    """
    tasks = list()
    tasks.append(LoadData(header=kwargs.get('header'),
                          resolution=kwargs.get('resolution', 3.)))
    if 'compression' in kwargs:
        tasks.append(CompressData(kwargs.get('compression'),
                                  kwargs.get('factor', 1.)))
    if 'deglitching_filter_length' in kwargs:
        tasks.append(Deglitching(kwargs.get('deglitching_filter_length')))
    if 'noise_filter_length' in kwargs:
        tasks.append(NoiseFiltering(kwargs.get('noise_filter_length')))
    tasks.append(AsLinearOperator())
    # should be added if compression is present !
    if 'compression_model' in kwargs:
        tasks.append(CompressionModel(kwargs.get('compression_model', 'no')))
    if 'map_mask' in kwargs:
        if kwargs.get('map_mask'):
            tasks.append(MapMaskModel())
    if 'algo' in kwargs:
        if 'hypers' in kwargs:
            tasks.append(DefineSmoothnessPriors())
        tasks.append(Inversion(kwargs.pop('algo'), **kwargs))
    return Pipeline(tasks, verbose=kwargs.get('verbose', False))

def compress(tod, C, factor):
    ctod = (C * tod.flatten())
    ctod = ctod.reshape((tod.shape[0] ,tod.shape[1] / factor))
    ctod = tm.Tod(ctod, mask=np.zeros(ctod.shape, dtype=np.int8))
    ctod.nsamples = [ns / factor for ns in tod.nsamples]
    return ctod

def uncompress(ctod, C, factor):
    uctod = tm.Tod(np.zeros((ctod.shape[0], ctod.shape[1] * factor)))
    y0, t = lo.sparse.spl.cgs(C.T * C, C.T * ctod.flatten())
    uctod[:] = y0.reshape(uctod.shape)
    return uctod

def load_data(filename, header=None, resolution=3.):
    #mask = np.zeros((32, 64), dtype=np.int8)
    obs = tm.PacsObservation(filename=filename,
                             fine_sampling_factor=1,
                             #detector_mask=mask
                             )
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
