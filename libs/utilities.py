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
import pydicom._storage_sopclass_uids

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


def writeDicom(dcmFileName, dcmImg):
    '''
    
    Description: Write empty Dicom file
    
    Input:
        - dcmFileName = File name, e.g. "myDicom.dcm".
        - dcmImg = image np array
    
    Output:
        - 
            
    
    Source:
    
    '''
    
    dcmImg = dcmImg.astype(np.uint16)

    # print("Setting file meta information...")

    # Populate required values for file meta information
    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  
    
    ds = pydicom.Dataset()
    ds.file_meta = meta
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"
    
    ds.Modality = "MR"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15
    
    ds.ImagesInAcquisition = "1"
    
    ds.Rows = dcmImg.shape[0]
    ds.Columns = dcmImg.shape[1]
    ds.InstanceNumber = 1
    
    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    
    # print("Setting pixel data...")
    ds.PixelData = dcmImg.tobytes()
    
    ds.save_as(dcmFileName)
    
    return


def writeDicomFromTemplate(dcmFileName, dcmImg, dcmH):
    '''
    
    Description: Write Dicom file from dicom header template
    
    Input:
        - dcmFileName = File name, e.g. "myDicom.dcm".
        - dcmImg = image np array
    
    Output:
        - 
            
    
    Source:
    
    '''
    
    dcmImg = dcmImg.astype(np.uint16)

    # print("Setting file meta information...")

    # Populate required values for file meta information
    meta = pydicom.Dataset()
    
    meta.MediaStorageSOPClassUID = dcmH.file_meta.MediaStorageSOPClassUID
    meta.MediaStorageSOPInstanceUID = dcmH.file_meta.MediaStorageSOPInstanceUID
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian 
    
    ds = pydicom.Dataset()
    ds.file_meta = meta
    
    # ViewPosition
    ds.add_new((0x0018,0x5101),'CS', dcmH.ViewPosition)    
    # BodyPartThickness
    ds.add_new((0x0018,0x11A0),'FL', dcmH.BodyPartThickness)
    # CompressionForce
    ds.add_new((0x0018,0x11A2),'FL', dcmH.CompressionForce)
    # ExposureTime
    ds.add_new((0x0018,0x1150),'FL', dcmH.ExposureTime)
    # XrayTubeCurrent
    ds.add_new((0x0018,0x1151),'FL', dcmH.XRayTubeCurrent)
    # Exposure
    ds.add_new((0x0018,0x1152),'FL', dcmH.Exposure)
    # ExposureInuAs
    ds.add_new((0x0018,0x1153),'FL', dcmH.ExposureInuAs)
    # kvP
    ds.add_new((0x0018,0x0060),'FL', dcmH.KVP)
    
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.SOPClassUID = dcmH.SOPClassUID
    ds.PatientName = dcmH.PatientName
    ds.PatientID = dcmH.PatientID
    
    ds.PatientAge = dcmH.PatientAge
    ds.ImagerPixelSpacing = dcmH.ImagerPixelSpacing
    
    ds.Modality = dcmH.Modality
    ds.SeriesInstanceUID = dcmH.SeriesInstanceUID
    ds.StudyInstanceUID = dcmH.StudyInstanceUID
    ds.FrameOfReferenceUID = dcmH.FrameOfReferenceUID
    
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15
    
    ds.ImagesInAcquisition = dcmH.ImagesInAcquisition
    ds.Manufacturer = dcmH.Manufacturer
    ds.DetectorElementSpacing = dcmH.DetectorElementSpacing
    ds.ImagerPixelSpacing = dcmH.ImagerPixelSpacing
    ds.PresentationIntentType = dcmH.PresentationIntentType
    ds.ImageLaterality = dcmH.ImageLaterality
    
    ds.ViewCodeSequence = dcmH.ViewCodeSequence
    
    ds.Rows = dcmImg.shape[0]
    ds.Columns = dcmImg.shape[1]
    ds.InstanceNumber = dcmH.InstanceNumber
    
    ds.ImageType = dcmH.ImageType
    
    ds.RescaleIntercept = dcmH.RescaleIntercept
    ds.RescaleSlope = dcmH.RescaleSlope

    ds.PhotometricInterpretation = dcmH.PhotometricInterpretation
    ds.PixelRepresentation = dcmH.PixelRepresentation
    
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    
    # print("Setting pixel data...")
    ds.PixelData = dcmImg.tobytes()
    
    ds.save_as(dcmFileName)  
    
    return


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