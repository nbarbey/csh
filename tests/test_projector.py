#!/usr/bin/env python
"""Test the validity of transpose of projector
"""

import os
import numpy as np
import csh
import tamasis

# define data set
datadir = os.path.join(tamasis.tamasis_dir, 'tests',)
filenames = os.path.join(datadir, 'frames_blue.fits[0:16]')
# no compression
# compression modes
compressions = ["",]
# median filter length
filtering = True
filter_length = 10000
hypers = (1e15, 1e15)
ext = ".fits"
pre = "ngc6946_rls_cov_"
# to store results
models = []
# to output only the model !!!
model_only = True
for comp in compressions:
    models.append(csh.rls(filenames, compression=comp, hypers=hypers,
                          deglitch=False,
                          filtering=filtering, filter_length=filter_length,
                          model_only=model_only
                          ))
for model in models:
    M = model.todense()
    MT = model.T.todense()
    assert np.all(M.T == MT)
