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

from libs.utilities import makedir, filesep, writeDicom
from libs.methods import get_XYZ_calc_positions, get_breast_masks, process_dense_mask, \
    get_calc_cluster, get_XYZ_cluster_positions, get_projection_cluster_mask, \
    apply_mtf_mask_projs

#%%

if __name__ == '__main__':
    
    number_calc = 8
    
    cluster_dimensions  = (14,14,14)        # In mm
    calc_dimensions     = (5.6,5.6,5.6)     # In mm
    
    cluster_pixel_size = 0.05               # In mm
    
    
    pathPatientCases            = '/home/rodrigo/Downloads/mc_insert'
    pathCalcifications          = '/media/rodrigo/Data/images/UPenn/Phantom/VCT/db_calcium/calc'
    pathCalcificationsReport    = '/media/rodrigo/Data/images/UPenn/Phantom/VCT/db_calcium/report.xlsx'
    pathMatlab                  = '/usr/local/R2019a/bin/matlab'
    pathLibra                   = 'LIBRA-1.0.4'
    pathAuxLibs                 = 'libs'
    pathBuildDirpyDBT           = '/home/rodrigo/Documents/rodrigo/codes/pyDBT/build'
    pathMTF                     = 'data/mtf_function_ffdm_pristina_fourier.npy'
    pathPatientDensity          = pathPatientCases + '/density'
    pathPatientCalcs            = pathPatientCases + '/calcifications'
    
    # Flags
    flags = dict()
    flags['fix_compression_paddle'] = False
    flags['print_debug'] = True
    flags['vct_image'] = False
    flags['delete_masks_folder'] = False
    flags['force_libra'] = False
    
    cluster_size = [int(x/cluster_pixel_size) for x in cluster_dimensions]
    calc_window  = [int(x/cluster_pixel_size) for x in calc_dimensions]
     
    contrasts = [0.35, 0.25]
    # for x in range(14):
    #     contrasts.append(0.85 * contrasts[x])
    
    # List all patients    
    patient_cases = [str(item) for item in pathlib.Path(pathPatientCases).glob("*") if pathlib.Path(item).is_dir()]
    
    makedir(pathPatientDensity)
    makedir(pathPatientCalcs)
    
    for patient_case in patient_cases:
        
        exams = [str(item) for item in pathlib.Path(patient_case, 'DBT').glob("*") if pathlib.Path(item).is_dir() and 'density' not in str(item) and 'calcifications' not in str(item)]
        
        for exam in exams:  

            path2write_patient_name = "{}{}{}".format(pathPatientCalcs , filesep(), "/".join(exam.split('/')[-3:]))
            
            # Case already processed
            if makedir(path2write_patient_name):
                continue  

            print("Processing " + path2write_patient_name)                   
        
            #%%     
            
            # Get  X, Y and Z position for each calcification
            (x_calc, y_calc, z_calc), _ = get_XYZ_calc_positions(number_calc, cluster_size, calc_window, flags)
            
            # Load each calcification and put them on specified position
            roi_3D = get_calc_cluster(pathCalcifications, pathCalcificationsReport, number_calc, cluster_size, x_calc, y_calc, z_calc, flags)
            
    
            #%%
            
            dcmFiles = [str(item) for item in pathlib.Path(exam).glob("*.dcm")]
            
            # Run LIBRA
            mask_dense, mask_breast, bdyThick = get_breast_masks(dcmFiles, exam, pathPatientDensity, pathLibra, pathMatlab, pathAuxLibs, flags)
            
            # Process dense mask
            final_mask, flags = process_dense_mask(mask_dense, mask_breast, cluster_size, dcmFiles, flags)
            
            if flags['compression_paddle_found']:
                print(exam)
                continue
            
            del mask_dense, mask_breast
            
            #%%
            
            # Reconstruct the dense mask and find the coords for the cluster
            (x_clust, y_clust, z_clust), geo, libFiles, bound_X = get_XYZ_cluster_positions(final_mask, bdyThick, pathBuildDirpyDBT, flags)
            
            del final_mask
            
            #%% 
            
            cluster_2D = np.mean(roi_3D, axis=-1)
            cluster_2D = 255*(cluster_2D - cluster_2D.min())/(cluster_2D.max() - cluster_2D.min())
            
            if flags['right_breast']:
                x_clust2save = x_clust+bound_X
            else:    
                cluster_2D = np.fliplr(cluster_2D)
                x_clust2save = geo.nx - x_clust
                
            plt.imsave("{}{}cluster_{}x_{}y_{}z.png".format(path2write_patient_name, filesep(), x_clust2save, y_clust, z_clust), cluster_2D, cmap='gray')
            
            # Inserting cluster at position and projecting the cluster mask
            projs_masks = get_projection_cluster_mask(roi_3D, geo, x_clust, y_clust, z_clust, cluster_pixel_size, libFiles, flags)
            
            # Apply the fitted MTF on the mask projections
            projs_masks_mtf = apply_mtf_mask_projs(projs_masks, len(dcmFiles), pathMTF, flags)
            
            
            cropCoords_file = pathlib.Path('{}{}{}{}Result_Images{}cropCoords.npy'.format(pathPatientDensity , filesep(), "/".join(exam.split('/')[-3:]), filesep(), filesep()))
            if cropCoords_file.is_file():
                cropCoords = np.load(str(cropCoords_file))
                flags['mask_crop'] = True
                flags['cropCoords'] = cropCoords
            else:
                flags['mask_crop'] = False
                
                
            flags['calc_coords'] = (x_clust, y_clust, z_clust)
            
            np.save(path2write_patient_name + '{}flags'.format(filesep()), flags)

            for contrast in contrasts:
            
                path2write_contrast = "{}{}contrast_{:.3f}".format(path2write_patient_name , filesep(), contrast)
                
                makedir(path2write_contrast)
                
                for dcmFile in dcmFiles:
                    
                    dcmH = pydicom.dcmread(str(dcmFile))
                    
                    dcmData = dcmH.pixel_array.astype('float32').copy()
                    
                    if flags['mask_crop']:
                        dcmData = dcmData[cropCoords[0]:cropCoords[1], cropCoords[2]:cropCoords[3]]
                                    
                    if not flags['right_breast']:
                        dcmData = np.fliplr(dcmData)
                    
                    ind = int(str(dcmFile).split('/')[-1].split('_')[1]) - 1

                    tmp_mask = np.abs(projs_masks_mtf[:,:,ind])
                    tmp_mask = (tmp_mask - tmp_mask.min()) / (tmp_mask.max() - tmp_mask.min())
                    tmp_mask[tmp_mask > 0] *= contrast
                    tmp_mask = 1 - tmp_mask
                    
                    dcmData[:,bound_X:] = dcmData[:,bound_X:] * tmp_mask
                    
                    if not flags['right_breast']:
                        dcmData = np.fliplr(dcmData)
                                                
                    dcmFile_tmp = path2write_contrast + '{}{}'.format(filesep(), dcmFile.split('/')[-1])
                    
                    writeDicom(dcmFile_tmp, np.uint16(dcmData))
 
            
            
            
            
            
        



