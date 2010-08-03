#!/usr/bin/env python
import os
import csh

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67614]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67615]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
compressions = ["", "ca", "cs"]
factor=8
#compressions = ["ca"]
# median filter length
deglitch=True
covariance=False
filtering = False
filter_length = 10000
#hypers = (1e9, 1e9)
hypers = (1e0, 1e0)
ext = ".fits"
pre = "ngc6946_rls_no_filter"
# to store results
sol = []
# define same header for all maps
tod, projection, header, obs = csh.load_data(filenames)
# get the weight map
weights = projection.transpose(tod.ones(tod.shape))
weights.writefits(os.path.join(output_path, pre + 'weights' + ext))
del tod, projection, obs
# find a map for each compression and save it
for comp in compressions:
    sol.append(csh.rls(filenames, compression=comp, hypers=hypers, 
                       header=header, factor=factor,
                       deglitch=deglitch, covariance=covariance,
                       filtering=filtering, filter_length=filter_length))
    fname = os.path.join(output_path, pre + comp + ext)
    sol[-1].writefits(fname)
