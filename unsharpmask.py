from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.ndimage
import os
from astropy.io import fits 
from sofia_analysis import cubes

### taken/adapted from Susan's code at 
### github.com/seclark/RHT/blob/master/rht.py

# Performs a circle-cut of given diameter on inkernel.
# Outkernel is 0 anywhere outside the window.   
def circ_kern(diameter):
    assert diameter % 2
    r = diameter // 2 #int(np.floor(diameter/2))
    mnvals = np.indices((diameter, diameter)) - r
    rads = np.hypot(mnvals[0], mnvals[1])
    return np.less_equal(rads, r).astype(np.int)

# Unsharp mask. Returns subtracted data (background removed)
def umask(data, radius, smr_mask=None):
    assert data.ndim == 2
    kernel = circ_kern(2*radius+1)
    outdata = scipy.ndimage.filters.correlate(data, kernel) 
    # Correlation is the same as convolution here because kernel is symmetric 
    # Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(kernel)
    subtr_data = data - outdata/kernweight
    return subtr_data
    
    ### skip this ###
    # Convert to binary data
    #bindata = np.greater(subtr_data, 0.0)
    #if smr_mask is None:
    #    return bindata
    #else:
    #    return np.logical_and(smr_mask, bindata


def umask_cubes():
    path, dirs, files = os.walk('DR2W/processed/').next()
    for cubename in cubes:
        if 'GALFA_HI_RA+DEC_'+cubename+'_UnsharpMask_r=30.fits' in files:
            print(cubename,'already done!')
            continue
        print(cubename,'starting...')
        cube = fits.open('DR2W/GALFA_HI_RA+DEC_'+cubename+'.fits')
        for i in range(cube[0].data.shape[0]):                 
            cube[0].data[i] = umask(cube[0].data[i],30)
        cube.writeto('DR2W/processed/GALFA_HI_RA+DEC_'+cubename+'_UnsharpMask_r=30.fits')
        print(cubename,'done.')
