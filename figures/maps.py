#!/usr/bin/env python
import getopt, sys, os
import numpy as np
import pyfits
from pylab import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from mpl_toolkits.axes_grid import AxesGrid

# data
data_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output') + os.sep
fname_ext = [data_path + 'ngc6946_cross_robust_noc_photproj.fits',
             data_path + 'ngc6946_cross_robust_noc.fits',
             data_path + 'ngc6946_cross_robust_ca.fits',
             data_path + 'ngc6946_cross_robust_cs.fits',
             data_path + 'ngc6946_cross_robust.fits',
             ]
letters = ['a', 'b', 'c', 'd', 'e']
letters = ['(' + l + ')' for l in letters]
out_fname = data_path + 'ngc6946_cross_robust_maps_grid' + '.png'

# create grid
# figure
#plt.gray()
fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols=(3, 2),
                axes_pad=0.0,
                share_all=True,
                )
# loop on each map
for i, fname in enumerate(fname_ext):
    data = np.flipud(pyfits.fitsopen(fname)[0].data.T)
    if i == 0:
        data /= 8.
    data = data ** .25
    data[np.isnan(data)] = 0
    extent = [0., 192., 0., 192.]
    im = grid[i].imshow(data, extent=extent, interpolation="nearest")
    grid[i].text(10, 170, letters[i], fontsize=20, color="white")
    data_zoom = data[135:155, 80:100,]
    axins = zoomed_inset_axes(grid[i], 2, loc=3) # zoom = 6
    extent = [80., 100., 192. - 155., 192. - 135, ]
    im2 = axins.imshow(data_zoom, extent=extent, interpolation="nearest")
    im2.set_clim([data.min(), data.max()])
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    im.set_clim([0., .3])
    im2.set_clim([0., .15])
    mark_inset(grid[i], axins, loc1=2, loc2=4, fc="none", ec="0.5")

fontsize=20
posx = 10
posy = 170
grid[-1].text(posx, posy, "(a): PL", fontsize=fontsize, color="black")
posy -= 25
grid[-1].text(posx, posy, "(b): PLI", fontsize=fontsize, color="black")
posy -= 25
grid[-1].text(posx, posy, "(c): ACI", fontsize=fontsize, color="black")
posy -= 25
grid[-1].text(posx, posy, "(d): HCI", fontsize=fontsize, color="black")
posy -= 25
grid[-1].text(posx, posy, "(e): NOC", fontsize=fontsize, color="black")

grid[0].set_xticks(())
grid[0].set_yticks(())
plt.subplots_adjust(left=0., right=1., top=.99, bottom=.01, wspace=0., hspace=0.)
plt.show()
fig.savefig(out_fname)
