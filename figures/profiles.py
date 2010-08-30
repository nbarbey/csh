#!/usr/bin/env python
import os
from copy import copy
import numpy as np
import pylab as pl
import pyfits
from radial_profile import radial_profile
import scipy.ndimage as ndi

# coordinates og NGC 6946
center = [308.72, 60.153] # deg
#center = [308.76, 60.153] # deg # corrected 
pa = 52.5 # deg
incl = 46 # deg

# read data
path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output')
files = [#'ngc6946_rls_.fits',
         #'ngc6946_rls_ca.fits',
         #'ngc6946_rls_cs.fits', 
         #'ngc6946_rls_filt1e4.fits',
         #'ngc6946_rls_filt1e4ca.fits',
         #'ngc6946_rls_filt1e4cs.fits', 
         #'ngc6946_rls_no_filter.fits',
         #'ngc6946_rls_no_filterca.fits',
         #'ngc6946_rls_no_filtercs.fits', 
         #'ngc6946_huber_filt1e3.fits',
         #'ngc6946_huber_filt1e3ca.fits',
         #'ngc6946_huber_filt1e3cs.fits', 
         #'ngc6946_huber_filt1e4.fits',
         #'ngc6946_huber_filt1e4ca.fits',
         #'ngc6946_huber_filt1e4cs.fits',
         'ngc6946_cross_robust_noc_photproj.fits',
         'ngc6946_cross_robust_noc.fits',
         'ngc6946_cross_robust_ca.fits',
         'ngc6946_cross_robust_cs.fits',
         'ngc6946_cross_robust.fits',
         ]
# prepend path
filenames = [os.path.join(path, f) for f in files]

# select region of interest
#wfilename = os.path.join(path, 'ngc6946_rls_weights.fits')
#w = pyfits.fitsopen(wfilename)
#weights = w[0].data.T
#wmin, wmax = 65, 80
#roi = (wmin < weights) * (weights < wmax)
#labels, nlabels = ndi.measurements.label(roi)
# select biggest region
#region_size = [np.sum(labels == i) for i in xrange(1, nlabels)]
#biggest_index = np.where(region_size == np.max(region_size))[0][0] + 1
#roi = labels == biggest_index
roi = np.ones((192, 192))

# intiat list of radial profiles
phot_tables = []
pix_tables = []
p = []
datas = []
headers = []
for i, filename in enumerate(filenames):
    # read data
    p.append(pyfits.fitsopen(filename)[0])
    datas.append(np.ma.MaskedArray(p[-1].data.T, mask=(1 - roi)))
    if i==0:
        # backprojection normalize
        datas[-1] /= 8
    headers.append(p[-1].header)
    # run radial_profile
    radii, phot_table, pix_table = radial_profile(datas[-1], headers[-1],
                                                  center, incl, pa)
    phot_tables.append(phot_table)
    pix_tables.append(pix_table)

# compute background
bkg_roi = np.zeros(datas[0].shape, dtype=np.bool8)
bkg_roi[0:40, 0:50] = 1.
bkg_roi[0:50, 150:-1] = 1.
bkg_roi[150:-1, 150:-1] = 1.
bkg_roi[150:-1, 0:50] = 1.

bkgs = [np.mean(d[bkg_roi]) for d in datas]
errors = [np.std(d[bkg_roi]) for d in datas]

# remove background
phot_tables = [x - bkg for x, bkg in zip(phot_tables, bkgs)]
# error bars
yerrors = []
for err, pix_table in zip(errors, pix_tables):
    yerrors.append(err / np.sqrt(pix_table))

# plot
fsize = 20
params = {
    'axes.labelsize': fsize,
    'text.fontsize': fsize,
    'legend.fontsize': fsize,
    'xtick.labelsize': fsize,
    'ytick.labelsize': fsize,
    'text.usetex': True,
    #'figure.figsize': fig_size
    }
pl.rcParams.update(params)

L = 83
labels = ['PL', 'PLI', 'ACI', 'HCI', 'NOC']
labels = ['(' + l + ')' for l in labels]
markers = ['-o', '-v', '-s', '-p', '-*']
pl.figure()
#pl.title('Radial profile')
pl.subplot(2, 1, 1)
pl.title('(a)', fontsize=fsize)
for f, y, m, yerr in zip(labels, phot_tables, markers, yerrors):
    #yerr = np.asarray((yerr,) * len(y))
    #yerr[y - yerr <0] = y[y - yerr <0] - 1e-8
    ylower = np.maximum(1e-7, y - yerr)
    yerr_lower = y - ylower
    pl.yscale('log')
    #pl.errorbar(radii[:L] , y[:L], yerr=yerr, fmt=m, label=f)
    pl.errorbar(radii[:L] , y[:L],
                #yerr=np.log10(y[:L]) * yerr / y[:L],
                yerr=[yerr_lower[:L], yerr[:L]],
                fmt=m, label=f)
    pl.setp(pl.gca(), 'xticklabels', [])
    pl.ylabel('Surface brightness (Jy / pixel)')
    pl.ylim([1e-7, 1e-2])

leg = pl.legend(loc='upper right')

pl.subplot(2, 1, 2)
pl.title('(b)', fontsize=fsize)
for f, y, m in zip(labels, phot_tables, markers):
    pl.plot(radii[:L] , y[:L] / phot_tables[-1][:L], m, label=f)
    pl.xlabel('distance from the center (arcsec)')
    pl.ylabel('Surface brightness ratio')

pl.subplots_adjust(left=.1, right=.95, top=.95, bottom=.05, wspace=.1, hspace=.1)
pl.show()
pl.savefig(os.path.join(path, 'ngc6946_radial_profile.png'))

# compute flux
flux = [y * n for y, n in zip(phot_tables, pix_tables)]
total_flux = [np.sum(f) for f in flux]
pl.figure()
for y, m, f in zip(flux, markers, labels):
    pl.plot(radii[:-2], y[:-2], m, label=f)

pl.legend()
pl.savefig(os.path.join(path, 'ngc6946_radial_flux.png'))
