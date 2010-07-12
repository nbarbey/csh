#!/usr/bin/env python
import pyfits
from pylab import matplotlib
from matplotlib.pyplot import *
from numpy import *
import getopt, sys, os

fname_ext = sys.argv[1]
fname = fname_ext.split('.')[0]
out_fname = fname + '.png'
print('displaying ' + fname)
title_str = fname.split(os.sep)[-1]
t = pyfits.fitsopen(fname_ext)[0].data
h = figure()
#imshow(flipud(t.T) , interpolation="nearest")
#imshow(flipud(t.T) ** .25, interpolation="nearest")
imshow(flipud((t - t.min()).T) ** .25, interpolation="nearest")
colorbar()
title(title_str)
xlabel('Right Ascension')
ylabel('Declination')
show()
h.savefig(out_fname)
