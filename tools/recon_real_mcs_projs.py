#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:42:15 2022

@author: rodrigo
"""

import sys
import numpy as np
import pydicom
import pathlib
import matplotlib.pyplot as plt

from scipy.io import loadmat

sys.path.insert(1, '../')
sys.path.insert(1, '/home/rodrigo/Documents/Rodrigo/Codigos/pyDBT')

from libs.utilities import makedir, filesep, writeDicom

from pydbt.functions.projection_operators import backprojectionDDb_cuda
from pydbt.parameters.parameterSettings import geometry_settings
from pydbt.functions.initialConfig import initialConfig
from pydbt.functions.dataPreProcess import dataPreProcess

def readDicom(path, geo):
    
    dcmFiles = [str(item) for item in pathlib.Path(path).glob("*.dcm")]
    
    # Test if list is empty
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')
    
    proj = [None] * geo.nProj
    proj_header = [None] * geo.nProj
    
    for f in dcmFiles:
        nProj = int(f.split('/')[-1].split('_')[1]) - 1
        proj_header[nProj] = pydicom.dcmread(f)
        proj[nProj]  = proj_header[nProj].pixel_array
    
    proj = np.stack(proj, axis=-1).astype(np.uint16)
    
    return proj, proj_header

def recon_exam(geo, patient_case): 
    
    print('Processing case: {}'.format("/".join(patient_case.split('/')[-4:])))
    
    path2write_patient_name = path2write + "/" + "/".join(patient_case.split('/')[-4:])
    
    if not makedir(path2write_patient_name):
    
        proj, dcmH = readDicom(patient_case, geo)
        
        if dcmH[0].ImageLaterality == 'R':
            flags['right_breast'] = True
        elif dcmH[0].ImageLaterality == 'L':
            flags['right_breast'] = False
                
        flags['flip_projection_angle'] = False
        
        # Some projections start from positive DetectorSecondaryAngle, so we flip them 
        if dcmH[0].DetectorSecondaryAngle > 0:
            if not flags['right_breast']:
                flags['flip_projection_angle'] = True
             
        else:
            if flags['right_breast']:
                flags['flip_projection_angle'] = True
                
        if not flags['right_breast']:
            proj = np.fliplr(proj)
            
        if flags['flip_projection_angle']:
            proj = np.flip(proj, axis=-1)      
        
        proj = dataPreProcess(proj, geo)
    
        
        geo.nx = proj.shape[1]      # number of voxels (columns)
        geo.ny = proj.shape[0]      # number of voxels (rows)
        geo.nu = proj.shape[1]      # number of pixels (columns)
        geo.nv = proj.shape[0]      # number of pixels (rows)
        geo.nz = np.ceil(np.float32(dcmH[0].BodyPartThickness)/geo.dz).astype(int)
        
        vol = backprojectionDDb_cuda(np.float64(proj), geo, -1, libFiles)
        
        # vol = (vol - vol.min()) / (vol.max() - vol.min()) + 1e-5
        
        # vol = np.log(vol)
        
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        
        vol = vol * (2**12-1)
        
        vol = np.uint16(vol)           
        
        for z in range(vol.shape[-1]):
            
            dcmFile_tmp = path2write_patient_name + '{}{}.dcm'.format(filesep(), z)
            
            writeDicom(dcmFile_tmp, np.uint16(vol[:,:,z]))
        
    return


#%%

if __name__ == '__main__':
    
    pathPatientCases            = '/home/rodrigo/Dropbox/calc_files'
    pathBuildDirpyDBT           = '/home/rodrigo/Documents/Rodrigo/Codigos/pyDBT/build'
    path2write                  = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/Imagens_Clinicas_Pristina_Out-2021_(Organizado_Bi-Rads)/recons_LAVI'
    
    # Flags
    flags = dict()
    
    f =  open(pathPatientCases, 'r') 
    patient_cases = f.read()
    f.close()
    patient_cases = patient_cases.replace(" ", "")
    patient_cases = patient_cases.split('\n')
    

    # Call function for initial configurations
    libFiles = initialConfig(buildDir=pathBuildDirpyDBT, createOutFolder=False)
    
    # Create a DBT geometry  
    geo = geometry_settings()
    geo.GE()
    
    for patient_case in patient_cases:
        
        
        if 'DBT' in patient_case:
                        
            recon_exam(geo, patient_case)
            
            if 'CC' in patient_case:
                
                patient_orientation = patient_case.split('/')[-1].split('_')[0]
                
                dirs = [str(item) for item in pathlib.Path("/".join(patient_case.split("/")[:-1])).glob("*") if item.is_dir() and (patient_orientation + '_MLO') in str(item)][0]
                
                recon_exam(geo, dirs)
            
            
            
            
                
                
