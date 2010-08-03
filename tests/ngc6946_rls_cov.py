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
#compressions = ["ca"]
# median filter length
deglitch=True
covariance=True
filtering = True
filter_length = 10000
hypers = (1e9, 1e9)
ext = ".fits"
pre = "ngc6946_rls_cov_"
# to store results
sol = []
# define same header for all maps
tod, projection, header, obs = csh.load_data(filenames[0])
del tod, projection, obs
# find a map for each compression and save it
for comp in compressions:
    sol.append(csh.rls(filenames, compression=comp, hypers=hypers, 
                       header=header,
                       deglitch=deglitch, covariance=covariance,
                       filtering=filtering, filter_length=filter_length))
    fname = os.path.join(output_path, pre + comp + ext)
    sol[-1].writefits(fname)
