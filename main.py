#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:27:29 2022

@author: rodrigo

This code does the whole thing - it generates artificial clusters,
finds candidate positions depending on the breast density and
insert the lesion in a clinical case.

OBS: This code uses LIBRA to estimate density. Please refer to
https://www.pennmedicine.org/departments-and-centers/department-of-radiology/radiology-research/labs-and-centers/biomedical-imaging-informatics/cbig-computational-breast-imaging-group
for more information.
 
"""


import sys
import numpy as np
import pydicom
import pathlib
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/rodrigo/Documents/rodrigo/codes/pyDBT')

from pydbt.functions.dataPreProcess import dataPreProcess
from pydbt.functions.projectionDD import projectionDD

from libs.utilities import makedir, filesep, writeDicom
from libs.methods import get_XYZ_calc_positions, get_breast_masks, process_dense_mask, \
    get_calc_cluster, get_XYZ_cluster_positions

#%%

if __name__ == '__main__':
    
    number_calc = 8
    
    cluster_dimensions  = (14,14,14)        # In mm
    calc_dimensions     = (5.6,5.6,5.6)     # In mm
    
    cluster_pixel_size = 0.05               # In mm
    
    
    pathPatientCases            = '/home/rodrigo/Downloads/Bi-Rads_1/1729110/DBT'#'/media/rodrigo/Data/images/UPenn/Phantom/VCT/VCT_Bruno_500/GE-projs'
    pathCalcifications          = '/media/rodrigo/Data/images/UPenn/Phantom/VCT/db_calcium/calc'
    pathCalcificationsReport    = '/media/rodrigo/Data/images/UPenn/Phantom/VCT/db_calcium/report.xlsx'
    pathMatlab                  = '/usr/local/R2019a/bin/matlab'
    pathLibra                   = 'LIBRA-1.0.4'
    pathBuildDirpyDBT           = '/home/rodrigo/Documents/rodrigo/codes/pyDBT/build'
    pathPatientDensity          = pathPatientCases + '/density'
    pathPatientCalcs            = pathPatientCases + '/calcifications'
    
    makedir(pathPatientDensity)
    makedir(pathPatientCalcs)
    
    cluster_size = [int(x/cluster_pixel_size) for x in cluster_dimensions]
    calc_window  = [int(x/cluster_pixel_size) for x in calc_dimensions]
     
    contrasts = [0.1]
    for x in range(14):
        contrasts.append(0.85 * contrasts[x])
    
    # List all patients    
    patient_cases = [str(item) for item in pathlib.Path(pathPatientCases).glob("*") if pathlib.Path(item).is_dir()]
    
    for patient_case in patient_cases:
        
        #%%     
        
        # Get  X, Y and Z position for each calcification
        (x_calc, y_calc, z_calc), _ = get_XYZ_calc_positions(number_calc, cluster_size, calc_window)
        
        # Load each calcification and put them on specified position
        roi_3D = get_calc_cluster(pathCalcifications, pathCalcificationsReport, number_calc, cluster_size, x_calc, y_calc, z_calc)
        

        #%%
        
        dcmFiles = [str(item) for item in pathlib.Path(patient_case).glob("*.dcm")]
        
        # Run LIBRA
        mask_dense, mask_breast, bdyThick = get_breast_masks(dcmFiles, patient_case, pathPatientDensity, pathLibra, pathMatlab)
        
        # Process dense mask
        final_mask, flag_right_breast = process_dense_mask(mask_dense, mask_breast, cluster_size)
        
        del mask_dense, mask_breast
        
        #%%
        
        # Reconstruct the dense mask and find the coords for the cluster
        (x_clust, y_clust, z_clust), geo, libFiles, bound_X = get_XYZ_cluster_positions(final_mask, bdyThick, pathBuildDirpyDBT)
        
        del final_mask
        
#%% 

        """
        Here, we are recreating the volume but now with higher resolution on
        the Z axis.      
        """
        
        print("Inserting cluster at position and projecting the cluster mask...")

        # We add an offset to the airgap so we dont need to project the
        # slices that dont have information.
        vol_z_offset = (z_clust * geo.dz) - (roi_3D.shape[2]//2*cluster_pixel_size)
        
        geo.x_offset = x_clust * geo.dx 
        geo.y_offset = (np.arange(-geo.ny/2, (geo.ny/2)+1, 1) * geo.dy)[y_clust]
        
        # We refresh the geometry parameters
        geo.nx = roi_3D.shape[0]
        geo.ny = roi_3D.shape[1]
        geo.nz = roi_3D.shape[2]
        
        geo.dx = cluster_pixel_size
        geo.dy = cluster_pixel_size
        geo.dz = cluster_pixel_size
        
        geo.DAG += vol_z_offset
        
        # Create a empty volume specific for that part (calcification cluster)
        vol = np.zeros((geo.ny, geo.nx, geo.nz))
        
        # Load the cluster on this volume. Note that Z has the same size
        vol = roi_3D
        
        # Project this volume
        projs_masks = projectionDD(np.float64(vol), geo, libFiles)
        
        
        for contrast in contrasts:
        
            path2write_patient_name = "{}{}{}-contrast{:.3f}".format(pathPatientCalcs , filesep(), patient_case.split('/')[-1], contrast)
            
            makedir(path2write_patient_name)
            
            for idX, dcmFile in enumerate(dcmFiles):
                
                dcmH = pydicom.dcmread(str(dcmFile))
                
                dcmData = dcmH.pixel_array.astype('float32').copy()
                                
                if not flag_right_breast:
                    dcmData = np.fliplr(dcmData)
                
                ind = int(str(dcmFile).split('/')[-1].split('_')[1]) - 1
                            
                tmp_mask = np.abs(projs_masks[:,:,ind])
                tmp_mask = (tmp_mask - tmp_mask.min()) / (tmp_mask.max() - tmp_mask.min())
                tmp_mask[tmp_mask > 0] *= contrast
                tmp_mask = 1 - tmp_mask
                
                dcmData[:,bound_X:] = dcmData[:,bound_X:] * tmp_mask
                
                if not flag_right_breast:
                    dcmData = np.fliplr(dcmData)
                            
                # dcmH.PixelData = np.uint16(dcmData).copy()
                
                dcmFile_tmp = path2write_patient_name + '{}{}'.format(filesep(), dcmFiles[idX].split('/')[-1])
                
                writeDicom(dcmFile_tmp, np.uint16(dcmData))
                
                # pydicom.dcmwrite(dcmFile_tmp,
                #                  dcmH, 
                #                  write_like_original=False) 
            
            
            
            
            
        



