#!/usr/bin/env python
import numpy as np
from copy import copy

def radial_profile(data, header, center=[0., 0.], incl=0., pa=0., 
                   bkg=0., upto=0., n_pix_ring=1.):
    """
    Compute radial profile of a galaxy

    """
    # convert center coordinate into pixels
    max_diff = 1e-3
    center = rd2pix(header, center)
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    # informations on pixel shape
    if np.abs(np.abs(cdelt1) - np.abs(cdelt2)) > max_diff * np.abs(cdelt2):
        raise ValueError('CDELT1 and CDELT2 are not equal')
    pixel_size = get_pixel_size(cdelt1, cdelt2)
    pixel_surface = get_pixel_surface(cdelt1, cdelt2)
    # rotation of angle pa
    ima_maj, ima_min = get_rotated_coordinates(pa, center, data.shape)
    # maximum size along semi major and minor angles
    max_a_size = get_max_a_size(pa, data.shape, upto, pixel_size)
    max_b_size = get_max_b_size(max_a_size, incl)
    # get the maximal number of rings
    n_rings = get_n_rings(max_b_size, n_pix_ring)
    # b axis for each ring in pixel
    b_axis = n_pix_ring * (np.arange(n_rings) + 1)
    # radius table in arcsec
    radii = b_axis / np.cos(np.radians(incl)) * np.abs(cdelt1) * 3600
    # initial tables
    phot_table = np.zeros(n_rings)
    pix_table = np.zeros(n_rings)
    # ellipse image for checking purpose
    ellipse_image = np.zeros(data.shape)
    # force not seen data to 0
    data.data[data.mask == True] = 0.
    # to count number of pixels seen
    seen = (1 - data.mask)
    # create a mask for each ellipse
    for i_ring in xrange(n_rings):
        mask_image = (ima_maj / (b_axis[i_ring] / np.cos(np.radians(incl)))) ** 2 
        mask_image += (ima_min / b_axis[i_ring]) ** 2
        mask_image = mask_image < 1
        this_ring = np.where(mask_image == 1)
        image_ellipse_before = copy(mask_image)
        if i_ring != 0:
            image_ellipse_now = copy(mask_image)
            mask_image -= image_ellipse_before
        ellipse_image[this_ring] = i_ring
        this_ring_phot = data.data[this_ring]
        this_ring_seen = seen[this_ring]
        if len(this_ring_phot) != 0:
            pix_table[i_ring] = this_ring_seen.sum()
            phot_table[i_ring] = this_ring_phot.sum()
    # differentiate to have in ring instead of ellipse
    phot_table[:-1] -= phot_table[1:]
    pix_table[:-1] -= pix_table[1:]
    phot_table /= pix_table

    return radii, phot_table, np.abs(pix_table)

def rd2pix(header, center):
    """Convert ra dec coordinate into pixel coordinates
    """
    ra, dec = center
    # x0
    cdelt2 = header['CDELT2']
    crpix2 = header['CRPIX2']
    crval2 = header['CRVAL2']
    x0 = (dec - crval2) / cdelt2 + crpix2
    # Y0
    cdelt1 = header['CDELT1']
    crpix1 = header['CRPIX1']
    crval1 = header['CRVAL1']
    y0 = (ra - crval1) / cdelt1 + crpix1
    return x0, y0


def get_pixel_size(cdelt1, cdelt2):
    """Max of cdelt1 and cdelt2 in arcsec
    """
    return np.max([np.abs(cdelt1), np.abs(cdelt2)]) * 3600.

def get_pixel_surface(cdelt1, cdelt2):
    """Pixel surface in steradians
    """
    return np.abs(np.radians(cdelt1) * np.radians(cdelt2))

def get_rotated_coordinates(pa, center, shape):
    """Define images of rotated coordinates
    """
    pa = np.radians(pa)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    i = np.arange(shape[0]) - center[0]
    j = np.arange(shape[1]) - center[1]
    J, I = np.meshgrid(j, i)
    ima_maj = - I * cos_pa + J * sin_pa
    ima_min = I * sin_pa + J * cos_pa
    return ima_maj, ima_min

def get_max_a_size(pa, shape, upto, pixel_size):
    pa = np.radians(pa)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    max_a_size = np.min(shape) / (2 * np.max([cos_pa, sin_pa]))
    if upto != 0:
        max_a_size = np.min([max_a_size, upto / pixel_size])
    return max_a_size

def get_max_b_size(max_a_size, incl):
    return max_a_size * np.cos(np.radians(incl))

def get_n_rings(max_b_size, n_pix_ring):
    return int(np.floor(max_b_size / n_pix_ring))
