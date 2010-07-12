#!/usr/bin/env python
import getopt, sys, os
import numpy as np
import pyfits
from pylab import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset

#fname_ext = '/home/nbarbey/data/csh/output/ngc6946_cross_robust.fits'
fname_ext = sys.argv[1]
fname = fname_ext.split('.')[0]
out_fname = fname + '.png'
print('displaying ' + fname)
title_str = fname.split(os.sep)[-1]
t = np.flipud(pyfits.fitsopen(fname_ext)[0].data.T)
fig = plt.figure(1, [5,4])
ax = fig.add_subplot(111)

#imshow(t , interpolation="nearest")
#imshow((t - t.min())) ** .25, interpolation="nearest")
tt = t ** .25
tt[np.isnan(tt)] = 0
extent = [0., 192., 0., 192.]
ax.imshow(tt, extent=extent, interpolation="nearest")

tzoom = tt[135:155, 80:100,]
axins = zoomed_inset_axes(ax, 2, loc=3) # zoom = 6
extent = [80., 100., 192. - 155., 192. - 135, ]
im = axins.imshow(tzoom, extent=extent, interpolation="nearest")
im.set_clim([tt.min(), tt.max()])
plt.xticks(visible=False)
plt.yticks(visible=False)
#x1, x2, y1, y2 = 80., 100., 135., 155.,
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#plt.title(title_str)
#plt.colorbar()
#plt.xlabel('Right Ascension')
#plt.ylabel('Declination')
plt.show()
fig.savefig(out_fname)
