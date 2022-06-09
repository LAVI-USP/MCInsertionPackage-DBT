#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:56:42 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def array2D_from_array1D(y, kind='cubic'):
    '''
    Description: Create 2D matrix from 1D vector through interpolation.
    
    Input:
        - y = 1D array. It is assumed that this vector is symetric and odd. 
        - kind = Interpolation function used. See scipy docs
    
    Output:
        - 2D array
        
    Source: https://stackoverflow.com/a/37810928/8682939
    
    '''
    
    if y.size % 2 == 0:
        raise ValueError("Input must be odd")
    
    n = y.size // 2

    # A vector of distance (measured in pixels) from the center of vector V to each element of V
    r = np.expand_dims(np.hstack((np.arange(n,0,-1), np.arange(0,n+1,1))), axis=-1)

    # Now find the distance of each element of a square 2D matrix from it's centre. @(x,y)(sqrt(x.^2+y.^2)) is just the Euclidean distance function. `
    ri = np.sqrt(r**2+r.T**2)

    # Find indexes which are greater than the max distance
    idx_extra = ri>r[-1]

    # Truncate to max distance
    ri[idx_extra] = r[-1]

    # Find a function
    f = interp1d(r[n:,0], y[n:], kind=kind)

    img_2D = f(ri)

    img_2D[idx_extra] = 0
    
    return img_2D, f

# Values from NHS Breast Screening Programme Equipment Report
mtf = np.array((1, 0.84, 0.66, 0.47, 0.32, 0.22, 0.15, 0.11, 0.09, 0.09, 0.08))

# plt.plot(mtf)

mtf_2d, f = array2D_from_array1D(np.hstack((mtf[-1:0:-1],mtf)))

np.save("mtf_function_ffdm_pristina_fourier.npy", f)



