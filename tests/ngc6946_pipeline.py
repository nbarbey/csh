#!/usr/bin/env python
import os
import csh
import lo

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342185454_blue_PreparedFrames.fits[5954:67614]',
             datadir + '/1342185455_blue_PreparedFrames.fits[5954:67615]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
compression = csh.averaging
# define pipeline
pipeline = csh.get_pipeline(compression=compression,
                            compression_model=compression,
                            factor=8.,
                            header=None,
                            resolution=3.,
                            deglitching_filter_length=1000,
                            noise_filter_length=1000,
                            algo=lo.quadratic_optimization,
                            hypers=(1e-2,  1e-2),
                            maxiter=100,
                            tol=1e-6,
                            verbose=True
                            )
pipeline(filenames)
solution = pipeline.state['map']
solution.writefits(output_path + 'ngc6946.fits')
