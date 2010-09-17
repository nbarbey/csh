#!/usr/bin/env python
import getopt, sys, os
import numpy as np
import pyfits
from pylab import matplotlib
import matplotlib.pyplot as plt
import pywt
import lo
import star

# parameters
data_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output') + os.sep
fname = data_path + 'ngc6946_cross_robust.fits'
out_fname = data_path + 'ngc6946_cross_robust_haar' + '.png'
# data
data = np.flipud(pyfits.fitsopen(fname)[0].data.T)

# rescale data
data = data ** .25
data[np.isnan(data)] = 0

# haar transform
#W = lo.pywt_lo.wavelet2(data.shape, "haar", level=1)
W = lo.pywt_lo.wavedec(data.size, "haar", level=3)
#haar_map = (W * data.flatten()).reshape(data.shape)
haar_map = star.wavelet(data, "haar", 2)

# display
fig = plt.figure()
plt.imshow(haar_map, interpolation="nearest")
plt.show()
fig.savefig(out_fname)
