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
from pydbt.functions.dataPreProcess import dataPreProcess


#%%

if __name__ == '__main__':
    
    pathPatientCases            = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/mc_insert'
    pathBuildDirpyDBT           = '/home/rodrigo/Documents/Rodrigo/Codigos/pyDBT/build'
    pathPatientDensity          = pathPatientCases + '/density'
    pathPatientCalcs            = pathPatientCases + '/calcifications'
    
    cluster_dimensions  = (20, 20, 10)              # In mm
    
    contrasts = [0.3]
    
    # List all patients    
    patient_cases = [str(item) for item in pathlib.Path(pathPatientCalcs).glob("*") if pathlib.Path(item).is_dir()]
    
        
    # Call function for initial configurations
    libFiles = initialConfig(buildDir=pathBuildDirpyDBT, createOutFolder=False)
    
    # Create a DBT geometry  
    geo = geometry_settings()
    geo.GE()
    
    for patient_case in patient_cases:
        
        exams = [str(item) for item in pathlib.Path(patient_case, 'DBT').glob("*") if pathlib.Path(item).is_dir()]
        
        for exam in exams: 
            
            path2write_patient_name = "{}{}{}".format(pathPatientDensity , filesep(), "/".join(exam.split('/')[-3:]))
            
            res = loadmat('{}{}Result_Images{}Masks_{}.mat'.format(path2write_patient_name, filesep(), filesep(), 4))['res']
            
            mask_breast = res['BreastMask'][0][0]
                        
            bdyThick = np.load('{}{}Result_Images{}bodyPartThickness.npy'.format(path2write_patient_name, filesep(), filesep()))   
            
            path2write_patient_name = "{}{}{}".format(pathPatientCalcs , filesep(), "/".join(exam.split('/')[-3:]))
            
            flags =  np.load(path2write_patient_name + '{}flags.npy'.format(filesep()), allow_pickle=True)[()]
            
            if not flags['right_breast']:
                mask_breast = np.fliplr(mask_breast)

            bound_X = np.max((int(np.where(np.sum(mask_breast, axis=0) > 1)[0][0]) - 30, 0))

            for contrast in contrasts:
                
                path2write_contrast = "{}{}recon_contrast_{:.3f}_ROI".format(path2write_patient_name , filesep(), contrast)
                
                if makedir(path2write_contrast):
                    continue  
                
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
                
                dcmData = dataPreProcess(dcmData, geo,  flagCropProj=False)
                
                vol = backprojectionDDb_cuda(np.float64(dcmData), geo, -1, libFiles)
                
                vol[vol < 0 ] = 0
                
                vol = (vol / vol.max()) * (2**12-1)
                
                vol = np.uint16(vol)
                    
                
                # The cluster origin is located at the same as the DBT systemes, i.e., right midle. Z is at the half
                cluster_pixel_size = int(20/0.1)
                
                ind_x = ((flags['calc_coords'][0] + flags['bound_X']) - bound_X) - cluster_pixel_size
                ind_y = flags['calc_coords'][1] - (cluster_pixel_size // 2)
                ind_z = flags['calc_coords'][2]
                # plt.imshow(vol[ind_y:ind_y+cluster_pixel_size,
                #                 ind_x:ind_x+cluster_pixel_size,
                #                 flags['calc_coords'][2]], 'gray')
                
                path2write_contrast = "{}{}recon_contrast_{:.3f}_ROI".format(path2write_patient_name , filesep(), contrast)
                
                for z in range(-7,8):
                    
                    dcmFile_tmp = path2write_contrast + '{}{}.dcm'.format(filesep(), ind_z + z)
                    
                    writeDicom(dcmFile_tmp, np.uint16(vol[ind_y:ind_y+cluster_pixel_size, 
                                                          ind_x:ind_x+cluster_pixel_size,
                                                          ind_z + z]))
                    
                # if not flags['right_breast']:
                #     vol = np.fliplr(vol)
                    
                
                # path2write_contrast = "{}{}recon_contrast_{:.3f}".format(path2write_patient_name , filesep(), contrast)
                
                # makedir(path2write_contrast)
                
                # for z in range(vol.shape[-1]):
                    
                #     dcmFile_tmp = path2write_contrast + '{}{}.dcm'.format(filesep(), z)
                    
                #     writeDicom(dcmFile_tmp, np.uint16(vol[:,:,z]))
                    
                    
                
                
