#!/usr/bin/env python
import os
import csh

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67617]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67617]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
compressions = ["", "ca", "cs"]
# median filter length
filtering = True
filter_length = 3000
ext = ".fits"
pre = "ngc6946_rls_"
# to store results
sol = []
# find a map for each compression and save it
for comp in compressions:
    sol.append(csh.rls(filenames, compression=comp, 
                       filtering=filtering, filter_length=filter_length))
    fname = os.path.join(output_path, pre + comp + ext)
    sol[-1].writefits(fname)
