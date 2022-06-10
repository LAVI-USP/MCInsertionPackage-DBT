#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:17:42 2022

@author: rodrigo
"""

import numpy as np
import cv2
import pandas as pd
import zipfile
import pydicom
import subprocess
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import multivariate_normal

from .utilities import makedir, removedir, filesep

from pydbt.functions.phantoms import phantom3d
from pydbt.functions.projection_operators import backprojectionDDb_cuda, projectionDD
from pydbt.parameters.parameterSettings import geometry_settings
from pydbt.functions.initialConfig import initialConfig

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def get_projection_cluster_mask(roi_3D, contrasts_individual, geo, x_clust, y_clust, z_clust, cluster_pixel_size, libFiles, flags):
    
    """
    Here, we are recreating the volume but now with higher resolution on
    the Z axis.      
    """
    
    if flags['print_debug']:
        print("Inserting cluster at position and projecting the cluster mask...")
    
    # We add an offset to the airgap so we dont need to project the
    # slices that dont have information.
    vol_z_offset = (z_clust * geo.dz) - (roi_3D.shape[2]//2*cluster_pixel_size)
    
    geo.x_offset = ((geo.nx - 1) - x_clust) * geo.dx 
    geo.y_offset = (y_clust - (geo.ny/2)) * geo.dy
    
    # We refresh the geometry parameters
    geo.nx = roi_3D.shape[0]
    geo.ny = roi_3D.shape[1]
    geo.nz = roi_3D.shape[2]
    
    geo.dx = cluster_pixel_size
    geo.dy = cluster_pixel_size
    geo.dz = cluster_pixel_size
    
    geo.DAG += vol_z_offset
    
    
    projs_masks = np.zeros((geo.nv, geo.nu, geo.nProj))
    
    
    for idX, contrast_individual in enumerate(contrasts_individual):
        
        # Create a empty volume specific for that part (calcification cluster)
        vol = np.zeros((geo.ny, geo.nx, geo.nz))
        
        # Load the cluster on this volume. Note that Z has the same size
        vol = roi_3D[..., idX]
        
        # Project this volume
        projs_masks_tmp = projectionDD(np.float64(vol), geo, -1, libFiles)
        
        projs_masks_tmp = (projs_masks_tmp / projs_masks_tmp.max()) * contrast_individual
        
        projs_masks += projs_masks_tmp
        
        
    projs_masks /= projs_masks.max()
    
    
    if flags['flip_projection_angle']:
        projs_masks = np.flip(projs_masks, axis=-1)
    
    return projs_masks

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def get_XYZ_cluster_positions(final_mask, bdyThick, buildDir, flags):
    
    if flags['print_debug']:
        print("Reconstructing density mask and generate random coords for cluster...")
    
    # Call function for initial configurations
    libFiles = initialConfig(buildDir=buildDir, createOutFolder=False)
    
    # Create a DBT geometry  
    geo = geometry_settings()
    geo.GE()
    
    # Find the X boundary
    bound_X = int(np.where(np.sum(final_mask[:,:,4], axis=0) > 1)[0][0]) - 30
    
    # Crop to save reconstruction time
    final_mask = final_mask[:,bound_X:,:]
    
    geo.nx = final_mask.shape[1]      # number of voxels (columns)
    geo.ny = final_mask.shape[0]      # number of voxels (rows)
    geo.nu = final_mask.shape[1]      # number of pixels (columns)
    geo.nv = final_mask.shape[0]      # number of pixels (rows)
    geo.nz = np.ceil(bdyThick/geo.dz).astype(int)
    
    vol = backprojectionDDb_cuda(np.float64(final_mask), geo, -1, libFiles)
        
    # Avoid cluster on top or bottom 
    vol[:,:,-(geo.nz//4):] = 0
    vol[:,:,:(geo.nz//4)] = 0
            
    # Ramdomly selects one of the possible points
    i,j,k = np.where(vol>0.5)
    randInt = np.random.randint(0,i.shape[0])
    y_pos, x_pos, z_pos = (i[randInt],j[randInt],k[randInt])
    
    return (x_pos, y_pos, z_pos), geo, libFiles, bound_X

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def get_XYZ_calc_positions(number_calc, cluster_size, calc_window, flags):
    
    if flags['print_debug']:
        print("Generating XYZ positions for each calcification...")
    
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
        cluster_PDF[x_pos[calc_n]-np.ceil(calc_window[0]/2).astype(int):x_pos[calc_n]+np.floor(calc_window[0]/2).astype(int), 
                    y_pos[calc_n]-np.ceil(calc_window[1]/2).astype(int):y_pos[calc_n]+np.floor(calc_window[1]/2).astype(int),
                    z_pos[calc_n]-np.ceil(calc_window[2]/2).astype(int):z_pos[calc_n]+np.floor(calc_window[2]/2).astype(int)] *= microcalc_PDF
        
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
    index = np.min((np.max((index, 1+calc_window//2)), cluster_size - (calc_window//2)))
    
    return index

def gauss3D(roi_size, stdev):
    '''
    Source gauss fit : https://stackoverflow.com/a/25723181/8682939
    '''
    
    mu = [x // 2 for x in roi_size]
    
    xx, yy, zz = np.meshgrid(np.linspace(0,roi_size[0]-1,roi_size[0]), 
                             np.linspace(0,roi_size[1]-1,roi_size[1]),
                             np.linspace(0,roi_size[2]-1,roi_size[2]))
    
    xyz = np.column_stack([xx.flat, yy.flat, zz.flat])

    mean_gauss_2d = np.array([mu[0],mu[1],mu[2]])
    cov_gauss_2d = np.diagflat([stdev**2,stdev**2,stdev**2])

    w = multivariate_normal.pdf(xyz, mean=mean_gauss_2d, cov=cov_gauss_2d)
    
    w = w.reshape(xx.shape)
    
    w /= w.sum()
    
    return w


#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#


def get_calc_cluster(pathCalcifications, pathCalcificationsReport, number_calc, cluster_size, x_calc, y_calc, z_calc, flags):
    
    if flags['print_debug']:
        print("Loading each calcification and placing them at each position...")
    
    # Uncomment to use phantom calcification
    # calc_3d = phantom3d('',
    #                     n=10,
    #                     phantom_matrix=np.array([[1.0, 0.69, 0.92, 0.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    #   
    # calcs_3D = number_calc * [calc_3d]
    
    
    df = pd.read_excel(pathCalcificationsReport)
    
    df = df[df['Type'] == 'calc']
    df = df[df['BB_CountZ'] <= 10]
         
    rand_index = np.random.randint(0, df.shape[0], number_calc)
    
    cluster_size = np.hstack((cluster_size,np.array(number_calc)))
    
    roi_3D = np.zeros(cluster_size)
    
    contrasts_local = np.hstack((1, np.linspace(0.5,1,100,endpoint=False)[np.random.randint(0,99,number_calc-1)]))

    for idX, contrast in enumerate(contrasts_local):
        
        
        calc_size = (df.iloc[rand_index[idX]]['BB_CountZ'], 
                     df.iloc[rand_index[idX]]['BB_CountY'], 
                     df.iloc[rand_index[idX]]['BB_CountX'])
    
        calc_name = df.iloc[rand_index[idX]]['FileName'] + '_{}x{}x{}'.format(calc_size[2],
                                                                              calc_size[1],
                                                                              calc_size[0])
        
        # Extract zip in tmp folder
        zip_ref =  zipfile.ZipFile("{}/{}.zip".format(pathCalcifications,
                                                      calc_name),"r") 
        zip_ref.extractall("/tmp")
        
        # Read .raw file
        calc_3D = np.fromfile("/tmp/{}/{}.raw".format(calc_name,calc_name), dtype=np.uint8, sep="")
        
        # Reshape it 
        calc_3D = calc_3D.reshape(calc_size)
        
        # Fix dimensions (slice on last)
        calc_3D = np.transpose(calc_3D, (1, 2, 0))
        
        # Normalize by the sum of pixels equal to 1 on a 2D vertical projection
        calc_3D = (contrast / (np.sum(calc_3D, axis=-1).max()/255)) * (calc_3D / calc_3D.max())
        
        roi_3D[x_calc[idX]-(calc_3D.shape[0]//2):x_calc[idX]-(calc_3D.shape[0]//2)+calc_3D.shape[0],
               y_calc[idX]-(calc_3D.shape[1]//2):y_calc[idX]-(calc_3D.shape[1]//2)+calc_3D.shape[1],
               z_calc[idX]-(calc_3D.shape[2]//2):z_calc[idX]-(calc_3D.shape[2]//2)+calc_3D.shape[2], idX] =  calc_3D
    
    return roi_3D, contrasts_local

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def get_breast_masks(dcmFiles, patient_case, pathPatientDensity, pathLibra, pathMatlab, pathAuxLibs, flags):
    
        
        path2write_patient_name = "{}{}{}".format(pathPatientDensity , filesep(), "/".join(patient_case.split('/')[-3:]))
        
        if makedir(path2write_patient_name):
            flag_mask_found = True
        else:
            flag_mask_found = False
            
            if flags['print_debug']:
                print("Runing LIBRA to estimate density and breast mask...")
                
        if flags['print_debug']:
            print("Loading density and breast mask...")
            
        mask_dense = len(dcmFiles) * [None]
        mask_breast = len(dcmFiles) * [None]      

                
        for idX, dcmFile in enumerate(dcmFiles):
            
            ind = int(str(dcmFile).split('/')[-1].split('_')[1]) - 1
            
            if not flag_mask_found or flags['force_libra']:
            
                dcmH = pydicom.dcmread(str(dcmFile))
                                                
                # As we are using DBT, we need to change some header param
                dcmH.ImagesInAcquisition = '1'
                dcmH.Manufacturer = 'GE MEDICAL'
                
                if flags['vct_image']:
                    # Simulate random parameters for VCT data
    
                    # ViewPosition
                    dcmH.add_new((0x0018,0x5101),'CS', 'CC')
                    # BodyPartThickness
                    dcmH.add_new((0x0018,0x11A0),'DS', 60)
                    # CompressionForce
                    dcmH.add_new((0x0018,0x11A2),'DS', 119.5)
                    # ExposureTime
                    dcmH.add_new((0x0018,0x1150),'DS', 770)
                    # XrayTubeCurrent
                    dcmH.add_new((0x0018,0x1151),'DS', 100)
                    # Exposure
                    dcmH.add_new((0x0018,0x1152),'DS', 87)
                    # ExposureInuAs
                    dcmH.add_new((0x0018,0x1153),'DS', 86800)
                    # kvP
                    dcmH.add_new((0x0018,0x0060),'DS', 29)
                    
                else:
                    # BodyPartThickness
                    dcmH.add_new((0x0018,0x11A0),'FL', dcmH.BodyPartThickness)
                    # CompressionForce
                    dcmH.add_new((0x0018,0x11A2),'FL', dcmH.CompressionForce)
                    # ExposureTime
                    dcmH.add_new((0x0018,0x1150),'FL', dcmH.ExposureTime)
                    # XrayTubeCurrent
                    dcmH.add_new((0x0018,0x1151),'FL', dcmH.XRayTubeCurrent)
                    # Exposure
                    dcmH.add_new((0x0018,0x1152),'FL', dcmH.Exposure)
                    # ExposureInuAs
                    dcmH.add_new((0x0018,0x1153),'FL', dcmH.ExposureInuAs)
                    # kvP
                    dcmH.add_new((0x0018,0x0060),'FL', dcmH.KVP)
                
                                
                dcmFile_tmp = path2write_patient_name + '{}{}.dcm'.format(filesep(), ind)
                
                pydicom.dcmwrite(dcmFile_tmp,
                                 dcmH, 
                                 write_like_original=True) 
                
                
                subprocess.run("{} -r \"addpath(genpath('{}'));addpath('{}');run('libra_startup');libra('{}', '{}', 1);exit\" -nodisplay -nosplash".format(pathMatlab,
                                                              pathLibra,
                                                              pathAuxLibs,
                                                              dcmFile_tmp,
                                                              path2write_patient_name), shell=True)
        
            # Read masks from LIBRA
            res = loadmat('{}{}Result_Images{}Masks_{}.mat'.format(path2write_patient_name, filesep(), filesep(), ind))['res']
            
            mask_dense[ind] = res['DenseMask'][0][0]
            mask_breast[ind] = res['BreastMask'][0][0]
        
        
        if not flag_mask_found:
            bdyThick = np.float32(dcmH.BodyPartThickness)
            np.save('{}{}Result_Images{}bodyPartThickness'.format(path2write_patient_name, filesep(), filesep()), bdyThick)
        else:
            bdyThick = np.load('{}{}Result_Images{}bodyPartThickness.npy'.format(path2write_patient_name, filesep(), filesep()))   
        
        if flags['delete_masks_folder']:              
            removedir(path2write_patient_name)
                
        return mask_dense, mask_breast, bdyThick
    
#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def process_dense_mask(mask_dense, mask_breast, cluster_size, dcmFiles, flags):
    
    if flags['print_debug']:
        print("Processing density and breast mask...")
    
    mask_dense = np.stack(mask_dense, axis=-1)
    mask_breast = np.stack(mask_breast, axis=-1)
    
    indDcm = [idx for idx, dcmFile in enumerate(dcmFiles) if int(str(dcmFile).split('/')[-1].split('_')[1]) == 1][0]
    
    dcmH = pydicom.dcmread(str(dcmFiles[indDcm]))
    
    if dcmH.ImageLaterality:
        
        if dcmH.ImageLaterality == 'R':
            flags['right_breast'] = True
        elif dcmH.ImageLaterality == 'L':
            flags['right_breast'] = False
    else:
    
        if np.sum(mask_breast[:,0:10,0]) != 0:
            flags['right_breast'] = False
            
        else:
            flags['right_breast'] = True
            
    if not flags['right_breast']:
        
        mask_dense = np.fliplr(mask_dense)
        mask_breast = np.fliplr(mask_breast)
     
    flags['flip_projection_angle'] = False
    # Some projections start from positive DetectorSecondaryAngle, so we flip them 
    if dcmH.DetectorSecondaryAngle > 0:
        
        if not flags['right_breast']:
            flags['flip_projection_angle'] = True
         
    else:
        if flags['right_breast']:
            flags['flip_projection_angle'] = True
        
        
    if flags['flip_projection_angle']:
        mask_dense = np.flip(mask_dense, axis=-1)
        mask_breast = np.flip(mask_breast, axis=-1)
        
        
       
    flags['compression_paddle_found'] = False 
    if flags['fix_compression_paddle']:  
        
        g = np.array(((0 ,-1, 0),
                      (-1, 4,-1),
                      (0 ,-1, 0)))
    
        edges = cv2.filter2D(mask_breast[:,:,0], -1, g)
            
        lines = cv2.HoughLines(edges,1,np.pi/90,300)
        
        if lines is not None:
            n_halfpi_lines = np.sum([float(x).is_integer() for x in lines[:,0,1] / (np.pi/2)])
        else:
            n_halfpi_lines = 0
        
        if n_halfpi_lines >= 3:
            flags['compression_paddle_found'] = True
            
        elif n_halfpi_lines >= 1:
            
            img = 255* mask_breast[:,:,0]
            img = np.tile(np.expand_dims(img, axis=-1), (1,1,3))
            for line in lines:
                rho,theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            plt.imshow(img)   
    
    
    
    # Mask erosion to avoid regions too close to the skin, chest-wall and
    # pectoral muscle
    mask_breast[:,-1,:] = 0
    mask_breast[:,0,:] = 0    
    
    # Element for erosion
    element = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(cluster_size[0]//2,cluster_size[1]//2))
        
    # Element to removes isolated pixels
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
    
    final_mask = np.empty_like(mask_breast)
    
    for z in range(mask_dense.shape[-1]):
    
        # Mask erosion 
        mask_breast[:,:,z] = cv2.erode(mask_breast[:,:,z], element)
        
        # Removes isolated pixels
        clean_dense_mask = cv2.morphologyEx(mask_dense[:,:,z], cv2.MORPH_CLOSE, element1)
        clean_dense_mask = cv2.morphologyEx(clean_dense_mask, cv2.MORPH_OPEN, element2)
        clean_dense_mask = cv2.erode(clean_dense_mask, element2)
    
        mask_dense[:,:,z] = clean_dense_mask * mask_dense[:,:,z]        
    
            
    # Map of possible positions
    final_mask = mask_breast * mask_dense
    
    return final_mask, flags

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

def apply_mtf_mask_projs(projs_masks, n_projs, detector_size, pathMTF, flags):
    
    if flags['print_debug']:
        print("Applying MTF on MC masks...")
      
    # Load fiited MTF function
    f = np.load(pathMTF, allow_pickle=True)[()]
    
    nyquist = 1/(2*detector_size)
    
     
    # A vector of distance (measured in pixels) 
    x = np.linspace(nyquist, 0, projs_masks.shape[1]+1)
    
    if projs_masks.shape[1] % 2 == 0:
        x = np.hstack((x, x[1:-1][-1::-1]))
    else:
        x = np.hstack((x, x[0:-1][-1::-1]))
    
    
    y = np.linspace(nyquist, 0, projs_masks.shape[0]+1)
    
    if projs_masks.shape[0] % 2 == 0:
        y = np.hstack((y, y[1:-1][-1::-1]))
    else:
        y = np.hstack((y, y[0:-1][-1::-1]))
    

    xx, yy = np.meshgrid(x, y)
    
    # Now find the distance of each element of a square 2D matrix from it's centre
    ri = np.sqrt(xx**2+yy**2)
    
    # Find indexes which are greater than nyquist
    idx_extra = ri > nyquist
    
    # Truncate to max freq
    ri[idx_extra] = nyquist
    
    # Evaluate the fitted MTF function on the points
    mtf_2d = f(ri)
    
    mtf_2d = np.fft.ifftshift(mtf_2d)
    
    # mtf_2d = np.real(np.fft.ifft2(mtf_2d))
    
    # mtf_2d = mtf_2d / np.sqrt(np.sum(mtf_2d**2))
    
    # mtf_2d = np.abs(np.fft.fft2(mtf_2d))
    
    projs_masks_mtf = np.empty_like(projs_masks)
    
    pad_i = projs_masks.shape[0] 
    pad_j = projs_masks.shape[1] 
    
    for z in range(n_projs):                    
        
        # Pad projection space domain
        projs_mask_pad = np.pad(projs_masks[:,:,z], ((0,pad_i),(0,pad_j)))
                               
        # FFT of pad projection
        projs_mask_pad_fft = np.fft.fft2(projs_mask_pad) 
        
        # Get the modulus (we will use on the multiplication)
        projs_mask_pad_abs = np.abs(projs_mask_pad_fft)
        # Get the angle (we will keep the same)
        projs_mask_pad_angle = np.angle(projs_mask_pad_fft)
        
        # Multiplication of modulus
        projs_mask_pad_abs = projs_mask_pad_abs * mtf_2d
        
        # Transform polar to rectangular (modulus * np.exp(1j*angles))
        projs_mask_pad_fft = projs_mask_pad_abs * np.exp(1j*projs_mask_pad_angle)
        
        # iFFT
        projs_mask_pad = np.real(np.fft.ifft2(projs_mask_pad_fft))
        
        # Crop projection
        projs_masks_mtf[:,:,z] = projs_mask_pad[:pad_i, :pad_j] 
    
    return projs_masks_mtf

#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#