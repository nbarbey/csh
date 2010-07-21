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
# median filter length
filtering = True
filter_length = 10000
hypers = (1e8, 1e8)
ext = ".fits"
pre = "ngc6946_rls_cov_"
# to store results
sol = []
# find a map for each compression and save it
for comp in compressions:
    sol.append(csh.rls(filenames, compression=comp, hypers=hypers,
                       deglitch=False,
                       filtering=filtering, filter_length=filter_length,
                       algo=csh.lo.acg
                       ))
    fname = os.path.join(output_path, pre + comp + ext)
    sol[-1].writefits(fname)
