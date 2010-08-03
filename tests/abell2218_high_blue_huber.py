#!/usr/bin/env python
import os
import numpy as np
import csh
import lo

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342184598_blue_PreparedFrames.fits[5954:67614]',
             datadir + '/1342184599_blue_PreparedFrames.fits[5954:67615]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
factor = 8
#compressions = ["no", "ca", "cs", "dt"]
compressions = ["ca"]
# median filter length
deglitch=False
covariance=False
filtering = True
filter_length = 800
#hypers = (1e9, 1e9)
hypers = (1e1, 1e1)
#wavelet='haar'
wavelet=None
#deltas = (None, 1e-8, 1e-8, 1e-8)
deltas = (None, None, None)
ext = ".fits"
pre = "abell2218_high_blue_huber_filt8e2_"
# define same header for all maps
tod, projection, header, obs = csh.load_data(filenames)
# get the backprojection
bpj = projection.transpose(tod)
bpj.writefits(os.path.join(output_path, pre + 'bpj' + ext))
# get the weight map
weights = projection.transpose(tod.ones(tod.shape))
weights.writefits(os.path.join(output_path, pre + 'weights' + ext))
naive = bpj / weights
naive[np.isnan(naive)] = 0.
naive.writefits(os.path.join(output_path, pre + 'naive' + ext))
del tod, projection, obs
# find a map for each compression and save it
# to store results
bpj = True
sol = []
bpjs = []
for comp in compressions:
    print("Inversion with " + comp + " compression")
    if comp == "":
        hypers = (1/8., 1/8.)
    else:
        hypers = (1e0, 1e0)
    s, b = csh.rls(filenames, compression=comp, hypers=hypers, 
                       header=header, deltas=deltas,
                       deglitch=deglitch, covariance=covariance,
                       filtering=filtering, filter_length=filter_length,
                       algo=lo.hacg, tol=1e-8, wavelet=wavelet, bpj=bpj
                       )
    sol.append(s)
    bpjs.append(b)
    fname = os.path.join(output_path, pre + comp)
    bpjs[-1].writefits(fname + '_bpj' + ext)
    sol[-1].writefits(fname + ext)
