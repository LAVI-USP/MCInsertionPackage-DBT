#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:06:34 2022

@author: rodrigo


/home/rodrigo/Downloads/mc_insert/Out_2283824/DBT/L_CC
/home/rodrigo/Downloads/mc_insert/Out_90167381/DBT/L_CC
/home/rodrigo/Downloads/mc_insert/Out_2595542/DBT/L_CC
/home/rodrigo/Downloads/mc_insert/Out_465535/DBT/R_CC
/home/rodrigo/Downloads/mc_insert/Out_2253754/DBT/L_CC
/home/rodrigo/Downloads/mc_insert/Out_3163657/DBT/R_CC


/home/rodrigo/Downloads/mc_insert/Out_465535/DBT/R_MLO
/home/rodrigo/Downloads/mc_insert/Dez_90108398/DBT/4-L_MLO

# Right breast problem
/home/rodrigo/Downloads/mc_insert/Dez_2513604/DBT/4-L_MLO
/home/rodrigo/Downloads/mc_insert/Dez_2513604/DBT/5-L_MLO
/home/rodrigo/Downloads/mc_insert/Out_559693/DBT/L_MLO

"""

import pathlib
import numpy as np
import pydicom
import subprocess
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '/home/rodrigo/Documents/rodrigo/codes/pyDBT')
sys.path.insert(1, '../')

from libs.methods import get_breast_masks
    
from libs.utilities import makedir, filesep, writeDicomFromTemplate

files = ['/home/rodrigo/Downloads/mc_insert/Out_2283824/DBT/L_CC',
        '/home/rodrigo/Downloads/mc_insert/Out_90167381/DBT/L_CC',
        '/home/rodrigo/Downloads/mc_insert/Out_2595542/DBT/L_CC',
        '/home/rodrigo/Downloads/mc_insert/Out_465535/DBT/R_CC',
        '/home/rodrigo/Downloads/mc_insert/Out_2253754/DBT/L_CC',
        '/home/rodrigo/Downloads/mc_insert/Out_3163657/DBT/R_CC', # erro
        '/home/rodrigo/Downloads/mc_insert/Out_465535/DBT/R_MLO',
        '/home/rodrigo/Downloads/mc_insert/Dez_90108398/DBT/4-L_MLO',
        '/home/rodrigo/Downloads/mc_insert/Dez_2513604/DBT/4-L_MLO',
        '/home/rodrigo/Downloads/mc_insert/Dez_2513604/DBT/5-L_MLO',
        '/home/rodrigo/Downloads/mc_insert/Out_559693/DBT/L_MLO',]


pathPatientCases            = '/home/rodrigo/Downloads/mc_insert'
pathMatlab                  = '/usr/local/R2019a/bin/matlab'
pathLibra                   = '../LIBRA-1.0.4'
pathAuxLibs                 = '../libs'
pathPatientDensity          = pathPatientCases + '/density'
pathPatientCalcs            = pathPatientCases + '/calcifications'

flags = dict()
flags['fix_compression_paddle'] = False
flags['print_debug'] = True
flags['vct_image'] = False
flags['delete_masks_folder'] = False
flags['force_libra'] = False


for exam in files:
        
    dcmFiles = [str(item) for item in pathlib.Path(exam).glob("*.dcm")]
    
    # Get the wrong mask breast
    _, mask_breast, _ = get_breast_masks(dcmFiles, exam, pathPatientDensity, pathLibra, pathMatlab, pathAuxLibs, flags)
    
    mask_breast = np.stack(mask_breast, axis=-1)
    
    z = 4
    
    # Find the boundaries of the paddle
    # Height bounds
    mask_h = np.sum(mask_breast[:,:,z], 1) > 0
    res = np.where(mask_h == True)
    h_min, h_max = res[0][0], res[0][-1]
    
    h_min += 30
    h_max -= 30
    
    # Weight bounds
    mask_w = np.sum(mask_breast[:,:,z], 0) > 0
    res = np.where(mask_w == True)
    w_min, w_max = res[0][0], res[0][-1]
    
    path2write_patient_name = "{}{}{}".format(pathPatientDensity , filesep(), "/".join(exam.split('/')[-3:]))
        
    for idX, dcmFile in enumerate(dcmFiles):
        
        dcmH = pydicom.dcmread(str(dcmFile))
        
        ind = int(str(dcmFile).split('/')[-1].split('_')[1]) - 1
        
        if idX == 0:
            if dcmH.ImageLaterality == 'R':
                w_min += 30
            elif dcmH.ImageLaterality == 'L':
                w_max -= 30
                
            np.save('{}{}Result_Images{}cropCoords'.format(path2write_patient_name, filesep(), filesep()), (h_min, 
                                                                                                            h_max,
                                                                                                            w_min,
                                                                                                            w_max))
        
        dcmData = dcmH.pixel_array.astype('float32').copy()
        
        dcmData_crop = dcmData[h_min:h_max, w_min:w_max]
        
        # Still wrong
        if np.sum(dcmData_crop > 16380) > 0:
            
            print(dcmFile)
        
            plt.figure();plt.imshow(dcmData_crop[h_min:h_max, w_min:w_max], 'gray')
        
        else:
            
            dcmFile_tmp = path2write_patient_name + '{}{}.dcm'.format(filesep(), ind)
            
            dcmH.ImagesInAcquisition = '1'
            dcmH.Manufacturer = 'GE MEDICAL'
            dcmH.PixelRepresentation = 1
            
            # Write uncompressed Dicom with croped image
            writeDicomFromTemplate(dcmFile_tmp, 
                                    dcmData_crop,
                                    dcmH)
                   
            # Run LIBRA again
            subprocess.run("{} -r \"addpath(genpath('{}'));addpath('{}');run('libra_startup');libra('{}', '{}', 1);exit\" -nodisplay -nosplash".format(pathMatlab,
                                                          pathLibra,
                                                          pathAuxLibs,
                                                          dcmFile_tmp,
                                                          path2write_patient_name), shell=True)
            
        
        
    
    

    
    
