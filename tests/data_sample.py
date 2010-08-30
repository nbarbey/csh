#!/usr/bin/env python
import numpy as np
import pyfits
import os
import fht
import pylab as pl
from mpl_toolkits.axes_grid import AxesGrid

# data
datadir = os.getenv('CSH_DATA')
filename = datadir + '../lowfreq/1342182424_blue_level0Frames.fits'

data = pyfits.fitsopen(filename)[1].data[..., 50000:60000]
idx = 2000

im0 = data[..., idx].copy()
m = np.mean(data, axis=-1)
for i in xrange(data.shape[-1]):
    data[..., i] -= m

im = data[..., idx]
im_fht = fht.fht(im0)
im_fht[0, 0] = 0.

# figure
fig = pl.figure()
pl.gray()
grid = AxesGrid(fig, 111, nrows_ncols=(3, 1),
                axes_pad=0.0,
                share_all=True,
#                cbar_mode="each",
                )

ims = grid[0].imshow(im0, interpolation="nearest")
grid[0].text(2, 4, '(a)', fontsize=20, color="white")

grid[0].set_xticks(())
grid[0].set_yticks(())

ims = grid[1].imshow(im, interpolation="nearest")
grid[1].text(2, 4, '(b)', fontsize=20, color="white")
#grid.cbar_axes[0].colorbar(ims)

ims = grid[2].imshow(im_fht, interpolation="nearest")
grid[2].text(2, 4, '(c)', fontsize=20, color="white")
#grid.cbar_axes[1].colorbar(ims)

pl.show()
pl.savefig(os.getenv('HOME') + '/data/csh/output/hadamard_sample.png')
