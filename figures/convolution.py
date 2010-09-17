#!/usr/bin/env python
"""
Displays projection matrix
"""

import os
import copy
import numpy as np
import lo
import csh

nk = 3
#kernel = np.ones((nk, nk))
#kernel[kernel.shape[0] / 2, kernel.shape[1] / 2] *= 2
kernel = np.asarray([[.5, 1, .5], [1, 2, 1], [.5, 1., .5]])
n = 16
P = lo.convolve((n, n), kernel, mode="same")
Pd = P.todense()
# get the model dense matrix
PtPd = (P.T * P).todense()
