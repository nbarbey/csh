#!/usr/bin/env python
import os
import numpy as np
import pyfits

version = '1.0.3'
filename = os.path.join(os.getenv('HOME'), 'projets', 'tamasis-' + version,
                        'pacs', 'data', 'PCalPhotometer_BadPixelMask_FM_v5.fits')
# save a backup
os.system('cp ' + filename + ' ' + filename + '.bak')
# load file
fits = pyfits.fitsopen(filename)
# put zeros into data
for f in fits[1:]:
    f.data = np.zeros(f.data.shape)
# remove old file
os.system('rm -f ' + filename)
# save file
fits.writeto(filename)
