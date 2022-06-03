#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:37:32 2022

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


#%%

if __name__ == '__main__':
    
    pathPatientCases            = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/mc_insert'
    pathBuildDirpyDBT           = '/home/rodrigo/Documents/Rodrigo/Codigos/pyDBT/build'
    pathPatientDensity          = pathPatientCases + '/density'
    pathPatientCalcs            = pathPatientCases + '/calcifications'
    
    contrasts = [0.35, 0.25]
    
    # List all patients    
    patient_cases = [str(item) for item in pathlib.Path(pathPatientCases).glob("*") if pathlib.Path(item).is_dir()]
    
    # Call function for initial configurations
    libFiles = initialConfig(buildDir=pathBuildDirpyDBT, createOutFolder=False)
    
    # Create a DBT geometry  
    geo = geometry_settings()
    geo.GE()
    
    for patient_case in patient_cases:
        
        exams = [str(item) for item in pathlib.Path(patient_case, 'DBT').glob("*") if pathlib.Path(item).is_dir() and 'density' not in str(item) and 'calcifications' not in str(item)]
        
        for exam in exams: 
            
            path2write_patient_name = "{}{}{}".format(pathPatientDensity , filesep(), "/".join(exam.split('/')[-3:]))
            
            res = loadmat('{}{}Result_Images{}Masks_{}.mat'.format(path2write_patient_name, filesep(), filesep(), 4))['res']
            
            mask_breast = res['BreastMask'][0][0]
                        
            bdyThick = np.load('{}{}Result_Images{}bodyPartThickness.npy'.format(path2write_patient_name, filesep(), filesep()))   
            
            path2write_patient_name = "{}{}{}".format(pathPatientCalcs , filesep(), "/".join(exam.split('/')[-3:]))
            
            flags =  np.load(path2write_patient_name + '{}flags.npy'.format(filesep()), allow_pickle=True)[()]
            
            if not flags['right_breast']:
                mask_breast = np.fliplr(mask_breast)

            bound_X = int(np.where(np.sum(mask_breast, axis=0) > 1)[0][0]) - 30

            for contrast in contrasts:
                
                
                path2write_contrast = "{}{}contrast_{:.3f}".format(path2write_patient_name , filesep(), contrast)
    
                dcmFiles = [str(item) for item in pathlib.Path(path2write_contrast).glob("*.dcm")]
                
                dcmData = len(dcmFiles) * [None]
                
                for dcmFile in dcmFiles:
                    
                    dcmH = pydicom.dcmread(str(dcmFile), force=True)
                    
                    ind = int(str(dcmFile).split('/')[-1].split('_')[1]) - 1
                    
                    dcmData[ind] = dcmH.pixel_array.astype('float32').copy()
                 
                    
                dcmData = np.stack(dcmData, axis=-1) 
                       
                
                if not flags['right_breast']:
                    dcmData = np.fliplr(dcmData)
                    
                if flags['flip_projection_angle']:
                    dcmData = np.flip(dcmData, axis=-1)
                 
                # Crop to save reconstruction time
                dcmData = dcmData[:,bound_X:,:]
                
                
                geo.nx = dcmData.shape[1]      # number of voxels (columns)
                geo.ny = dcmData.shape[0]      # number of voxels (rows)
                geo.nu = dcmData.shape[1]      # number of pixels (columns)
                geo.nv = dcmData.shape[0]      # number of pixels (rows)
                geo.nz = np.ceil(bdyThick/geo.dz).astype(int)
                
                vol = backprojectionDDb_cuda(np.float64(dcmData), geo, -1, libFiles)
                
                vol = np.uint16(vol)
                
                # plt.imshow(vol[:,:,flags['calc_coords'][-1]], 'gray')
                
                path2write_contrast = "{}{}recon_contrast_{:.3f}".format(path2write_patient_name , filesep(), contrast)
                
                makedir(path2write_contrast)
                
                for z in range(vol.shape[-1]):
                    
                    dcmFile_tmp = path2write_contrast + '{}{}.dcm'.format(filesep(), z)
                    
                    writeDicom(dcmFile_tmp, np.uint16(vol[:,:,z]))
                
                
