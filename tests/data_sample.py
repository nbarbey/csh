#!/usr/bin/env python
import pyfits
import os

# data
datadir = os.getenv('CSH_DATA')
filename = datadir + '../lowfreq/1342182424_blue_level0Frames.fits'

data = pyfits.fitsopen(filename)[1].data
