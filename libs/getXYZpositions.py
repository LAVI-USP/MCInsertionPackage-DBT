#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:15:53 2022

@author: rodrigo
"""

import numpy as np
from scipy.stats import multivariate_normal


def getXYZpositions(number_calc, cluster_size=(200,200,28), calc_window=(80,80,10)):
    
    x_pos = number_calc * [None]
    y_pos = number_calc * [None]
    z_pos = number_calc * [None]
    cluster_PDF_history = number_calc * [None]
        
    microcalc_PDF = gauss3D(calc_window, stdev=10)
    microcalc_PDF = 1 - ((microcalc_PDF - microcalc_PDF.min()) / (microcalc_PDF.max() - microcalc_PDF.min()))
    
    cluster_PDF = gauss3D(cluster_size, stdev=30)
        
    for calc_n in range(number_calc):
        
        proj_2D_PDF = np.sum(cluster_PDF, axis=-1)
        proj_2D_PDF /= proj_2D_PDF.sum()                    # Normalize PDF
        
        # Get x index
        x_pos[calc_n] = getIndex(proj_2D_PDF, cluster_size[0], calc_window[0])
        
        # Get y index
        y_pos[calc_n] = getIndex(proj_2D_PDF[:, x_pos[calc_n]:x_pos[calc_n]+1].T, cluster_size[1], calc_window[1])
        
        # Get z index
        z_pos[calc_n] = getIndex(cluster_PDF[y_pos[calc_n], x_pos[calc_n]:x_pos[calc_n]+1, :], cluster_size[2], calc_window[2])
        
        # Update cluster PDF, nocking out where we put the current calcification
        cluster_PDF[x_pos[calc_n]-(calc_window[0]//2):x_pos[calc_n]+calc_window[0]//2, 
                    y_pos[calc_n]-(calc_window[1]//2):y_pos[calc_n]+calc_window[1]//2,
                    z_pos[calc_n]-(calc_window[2]//2):z_pos[calc_n]+calc_window[2]//2] *= microcalc_PDF
        
        # Normalize for PDF
        cluster_PDF /= cluster_PDF.sum()
        
        # Store each cluster PDF
        cluster_PDF_history[calc_n] = cluster_PDF.copy()
        
    
    return (x_pos, y_pos, z_pos), cluster_PDF_history

def getIndex(cluster_PDF, cluster_size, calc_window):
    
    x_PDF = np.sum(cluster_PDF, axis=0)     # Get x-axis PDF
    x_PDF /= x_PDF.sum()                    # Normalize PDF
    
    x_CDF = np.cumsum(x_PDF)
    
    # Generate random probability 0-1
    x_rand = np.random.uniform()
    
    # Find the index of the corresponding probability in the CDF
    index = np.where((x_rand >= x_CDF) == False)[0][0]
    
    # Make sure the calcification window fits on the cluster window
    index = np.min((np.max((index, calc_window//2)), cluster_size - (calc_window//2)))
    
    return index

def gauss3D(roi_size, stdev):
    '''
    Source gauss fit : https://stackoverflow.com/a/25723181/8682939
    '''
    
    mu = [x // 2 for x in roi_size]
    
    xx, yy, zz = np.meshgrid(np.linspace(0,roi_size[0],roi_size[0]), 
                             np.linspace(0,roi_size[1],roi_size[1]),
                             np.linspace(0,roi_size[2],roi_size[2]))
    
    xyz = np.column_stack([xx.flat, yy.flat, zz.flat])

    mean_gauss_2d = np.array([mu[0],mu[1],mu[2]])
    cov_gauss_2d = np.diagflat([stdev**2,stdev**2,stdev**2])

    w = multivariate_normal.pdf(xyz, mean=mean_gauss_2d, cov=cov_gauss_2d)
    
    w = w.reshape(xx.shape)
    
    w /= w.sum()
    
    return w