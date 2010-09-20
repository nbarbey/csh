#!/usr/bin/env python
import getopt, sys, os
import numpy as np
import pyfits
from pylab import matplotlib
import matplotlib.pyplot as plt
import pywt
import lo

# data
data_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output') + os.sep
fname = data_path + 'ngc6946_cross_robust.fits'
out_fname = data_path + 'ngc6946_cross_robust_haar' + '.png'

# create grid
# figure
#plt.gray()
fig = plt.figure()

data = np.flipud(pyfits.fitsopen(fname)[0].data.T)

# haar transform
W = lo.pywt_lo.wavelet2(data.shape, "haar", level=3)
haar_map = (W * data.flatten()).reshape(data.shape)

#data = data ** .25
#data[np.isnan(data)] = 0


im = plt.imshow(data, extent=extent, interpolation="nearest")
im.set_clim([0., .3])

plt.show()
fig.savefig(out_fname)
