#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:15:03 2022

@author: rodrigo
"""

import os
import pydicom
import numpy as np
import pathlib

def filesep():
    """Check the system and use / or \\"""
    
    if os.name == 'posix':
        return '/'
    else:
        return '\\'
    

def makedir(path2create):
    """Create directory if it does not exists."""
 
    error = 1
    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0
    
    return error

def removedir(directory):
    """Link: https://stackoverflow.com/a/49782093/8682939"""
    directory = pathlib.Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            removedir(item)
        else:
            item.unlink()
    directory.rmdir()

def readDicom(dir2Read):
    """Read dicom folder."""
    
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))
    
    dcmData = len(dcmFiles) * [None]
    dcmHdr = len(dcmFiles) * [None]
    
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        
        dcmH = pydicom.dcmread(str(dcm))
           
        ind = int(str(dcm).split('/')[-1].split('.')[0][1:]) 
        
        dcmHdr[ind] = dcmH
                
        dcmData[ind] = dcmH.pixel_array.astype('float32')
        
    
    dcmData = np.stack(dcmData, axis=-1)
    
    return dcmData, dcmHdr

'''

for idX, k in enumerate(cluster_PDF_history):
    tmp = (k - k.min()) / (k.max() - k.min())
    tmp *= 65535
    os.mkdir('outputs_{}'.format(idX)) 
    for x in range(k.shape[-1]):
        plt.imsave('outputs_{}/{}.tiff'.format(idX, x), np.uint16(tmp[:,:,x]), cmap='gray', vmin=0, vmax=65535)

    
os.mkdir('outputs')  
k = ROI_3D.copy() 
k = (k - k.min()) / (k.max() - k.min())
k *= 65535
for z in range(ROI_3D.shape[-1]):
    plt.imsave('outputs/{}.tiff'.format(z), np.uint16(k[:,:,z]), cmap='gray', vmin=0, vmax=65535)
    
plt.imsave('proj.tiff'.format(z), np.uint16(np.mean(k, axis=-1)), cmap='gray')


os.mkdir('outputs')  
k = projs_masks.copy() 
k = (k - k.min()) / (k.max() - k.min())
k *= 65535
for z in range(projs_masks.shape[-1]):
    plt.imsave('outputs/{}.tiff'.format(z), np.uint16(k[:,:,z]), cmap='gray', vmin=0, vmax=65535)

'''