#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:38:18 2022

@author: rodrigo
"""

import sys
import numpy as np
import pydicom
import pathlib
import matplotlib.pyplot as plt

from scipy.io import loadmat

sys.path.insert(1, '../')

from libs.utilities import makedir, filesep, writeDicom


def readDicom(path):
    
    dcmFiles = [str(item) for item in pathlib.Path(path).glob("*.dcm")]
    
    # Test if list is empty
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')
    
    slices = [None] * len(dcmFiles)
    
    for f in dcmFiles:
        nSlice = int(f.split('/')[-1].split('.')[0])
        slices[nSlice] = pydicom.dcmread(f, force=True).pixel_array
    
    slices = np.stack(slices, axis=-1).astype(np.uint16)
    
    return slices

#%%

if __name__ == '__main__':
    
    pathPatientCases            = '/home/rodrigo/Dropbox/calc_files'
    path2read                   = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/Imagens_Clinicas_Pristina_Out-2021_(Organizado_Bi-Rads)/recons_LAVI'
    path2write                  = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/Imagens_Clinicas_Pristina_Out-2021_(Organizado_Bi-Rads)/recons_rois_LAVI'
    
    coords = [(1182, 1689, 80),
              (1043, 1619, 71),
              (1294, 997, 147),
              (1312, 602, 72),
              (1594, 451, 47),
              (1222, 410, 78),
              (1052, 574, 61),
              (1198, 1741, 36),
              (1108, 596, 51),
              (0, 0, 0),
              (1118, 497, 66),
              (1554, 514, 88),
              (1190, 1832, 9),
              (1145, 319, 16),
              (1546, 375, 43),
              (1157, 664, 64),
              (677, 802, 44),
              (990, 1185, 67),
              (1278, 837, 68),
              (1199, 1629, 15),
              (1148, 1777, 10),
              (1189, 1714, 13), # Bi-Rads_2/1760635/DBT/R_MLO_2
              (1339, 1782, 31),
              (1258, 568, 66),
              (1331, 584, 139),
              (1732, 568, 25),
              (1046, 538, 25), # Bi-Rads_2/1808158/DBT/R_MLO
              ]
    
    recon_dirs = [str(item) for item in pathlib.Path(path2read).glob("*/**") if item.is_dir() and ('CC' in str(item) or 'MLO' in str(item))]
    
    coord = (0,0)

    # for idX, (recon_dir, coord) in enumerate(zip(recon_dirs, coords)):
    for idX, recon_dir in enumerate(recon_dirs):
        
        print('Processing case: {}'.format("/".join(recon_dir.split('/')[-4:])))
        
        path2write_patient_name = path2write + "/" + "/".join(recon_dir.split('/')[-4:])
        
        if not makedir(path2write_patient_name):
            
            vol = readDicom(recon_dir)
            
            a = 1
            
            # coord = (1207, 345, 47)
            
            # plt.imshow(vol[coord[0]:coord[0]+200, coord[1]:coord[1]+200, coord[2]], 'gray')
            
            for z in range(-7,8):
                
                dcmFile_tmp = path2write_patient_name + '{}{}.dcm'.format(filesep(), coord[2] + z)
                
                writeDicom(dcmFile_tmp, np.uint16(vol[coord[0]:coord[0]+200, coord[1]:coord[1]+200, coord[2] + z]))
                

                '''
                coords = [(1182, 1689, 80),
                          (1043, 1619, 71),
                          (1294, 997, 147),
                          (1312, 602, 72),
                          (1594, 451, 47),
                          (1222, 410, 78),
                          (1052, 574, 61),
                          (1198, 1741, 36),
                          (1108, 596, 51),
                          (0, 0, 0),
                          (1118, 497, 66),
                          (1554, 514, 88),
                          (1190, 1832, 9),
                          (1145, 319, 16),
                          (1546, 375, 43),
                          (1157, 664, 64),
                          (677, 802, 44),
                          (990, 1185, 67),
                          (1278, 837, 68),
                          (1199, 1629, 15),
                          (1148, 1777, 10),
                          (1189, 1714, 13), # Bi-Rads_2/1760635/DBT/R_MLO_2
                          (1339, 1782, 31),
                          (1258, 568, 66),
                          (1331, 584, 139),
                          (1732, 568, 25),
                          (1586, 538, 25), # Bi-Rads_2/1808158/DBT/R_MLO
                          (1278, 1622, 66), # Bi-Rads_2/1870439/DBT/L_MLO
                          (1221, 323, 108),
                          (1274, 1740, 54), # Bi-Rads_2/2038867/DBT/R_CC
                          (1233, 758, 72), # Bi-Rads_2/2038867/DBT/R_MLO
                          (796, 818, 60), # 
                          (934, 914, 52), # Bi-Rads_2/2060247/DBT/L_MLO
                          (1245, 1179, 71), # Bi-Rads_2/2162013/DBT/L_MLO
                          (928, 755, 55), # Bi-Rads_2/2165900/DBT/L_CC
                          (1257, 627, 62), # Bi-Rads_2/2165900/DBT/L_MLO
                          (1097, 1519, 89), # Bi-Rads_2/2178692/DBT/L_MLO
                          (1619, 1564, 44), # Bi-Rads_2/2195734/DBT/L_CC
                          (1091, 277, 68), # Bi-Rads_2/2195734/DBT/L_MLO
                          (1051, 1727, 25), # Bi-Rads_2/2217424/DBT/L_CC
                          (924, 1585, 21), # Bi-Rads_2/2217424/DBT/L_MLO_1
                          (1388, 407, 69), # Bi-Rads_2/2241533/DBT/R_MLO
                          (1048, 1510, 17), # Bi-Rads_2/2246231/DBT/L_CC
                          (1393, 1649, 40), # Bi-Rads_2/2246231/DBT/L_MLO
                          (1254, 692, 32), # Bi-Rads_2/2258788/DBT/L_MLO
                          (0, 0, 0), # Bi-Rads_2/2265387/DBT/R_CC
                          (0, 0, 0), # Bi-Rads_2/2265387/DBT/R_MLO
                          (1506, 1719, 61), # Bi-Rads_2/2272806/DBT/L_CC
                          (1245, 1655, 66), # Bi-Rads_2/2272806/DBT/L_MLO
                          (1276, 1675, 39), # Bi-Rads_2/2272806/DBT/R_CC
                          (1106, 1675, 48), # Bi-Rads_2/2272806/DBT/R_MLO
                          (1006, 1534, 33), # Bi-Rads_2/2288763/DBT/R_CC
                          (1154, 1562, 41), # Bi-Rads_2/2288763/DBT/R_MLO_2
                          (1379, 1683, 43), # Bi-Rads_2/2324391/DBT/L_CC
                          (1436, 1782, 45), # Bi-Rads_2/2324391/DBT/L_MLO
                          (1045, 1263, 33), # Bi-Rads_2/2338519/DBT/L_CC   # Paciente com muita MC
                          (1045, 1266, 21), # Bi-Rads_2/2338519/DBT/L_MLO
                          (1087, 1407, 24), # Bi-Rads_2/2338519/DBT/R_CC
                          (1159, 1191, 59), # Bi-Rads_2/2338519/DBT/R_MLO
                          (1478, 528, 42), # Bi-Rads_2/2367793/DBT/L_CC
                          (1570, 542, 56), # Bi-Rads_2/2367793/DBT/L_MLO
                          (2006, 852, 53), # Bi-Rads_2/2367793/DBT/R_CC
                          (1699, 1079, 69), # Bi-Rads_2/2367793/DBT/R_MLO
                          (1668, 298, 81), # Bi-Rads_2/2375168/DBT/L_CC
                          (1348, 578, 71), # Bi-Rads_2/2375168/DBT/L_MLO
                          (1230, 1786, 54), # Bi-Rads_2/2435574/DBT/L_CC_2
                            (1205, 453, 59), # Bi-Rads_2/2435574/DBT/L_MLO
                            (1273, 660, 81), # Bi-Rads_2/2463088/DBT/L_MLO
                            (1556, 1825, 21), # Bi-Rads_2/2463187/DBT/R_CC
                            (0,0,0)
                            (1637, 665, 43), # Bi-Rads_2/2482750/DBT/L_CC
                            (1509, 626, 60), # Bi-Rads_2/2482750/DBT/L_MLO
                            (746, 538, 64), # Bi-Rads_2/2489465/DBT/L_CC
                            (766, 440, 55), # Bi-Rads_2/2489465/DBT/L_MLO_1
                            (1557, 376, 132), # Bi-Rads_2/2489465/DBT/R_CC
                            (1097, 399, 134), # Bi-Rads_2/2489465/DBT/R_MLO_1
                            (1332, 556, 37), # Bi-Rads_2/2548326/DBT/L_CC # Muita calc
                            (1352, 474, 67), # Bi-Rads_2/2548326/DBT/L_MLO # Muita calc
                            (1527, 500, 20), # Bi-Rads_2/2566045/DBT/L_CC
                            (0, 0, 0), # Bi-Rads_2/2566045/DBT/L_MLO
                            (1347, 925, 36), # Bi-Rads_2/2646039/DBT/R_CC
                            (1374, 888, 49), # Bi-Rads_2/2646039/DBT/R_MLO
                            (0, 0, 0), # Bi-Rads_2/2722805/DBT/L_CC_2
                            (0, 0, 0), # Bi-Rads_2/2722805/DBT/L_MLO
                            (1157, 462, 96), # Bi-Rads_2/478677/DBT/L_MLO
                            (1639, 469, 39), # Bi-Rads_2/478677/DBT/R_CC
                            (1720, 719, 55), # Bi-Rads_2/478677/DBT/R_MLO
                            (1414, 144, 85), # Bi-Rads_2/486871/DBT/L_CC
                            (917, 153, 95), # Bi-Rads_2/486871/DBT/L_MLO_1
                            (0,0,0), # Bi-Rads_2/486871/DBT/R_CC
                            (0,0,0), # Bi-Rads_2/486871/DBT/R_MLO
                            (996, 1660, 72), # Bi-Rads_2/498708/DBT/R_CC # Muita calc
                            (623, 363, 55), # Bi-Rads_2/498708/DBT/R_MLO
                            (1378, 276, 44), # Bi-Rads_2/561546/DBT/L_CC  # Muita calc
                            (1245, 272, 60), # Bi-Rads_2/561546/DBT/L_MLO
                            (1287, 282, 36), # Bi-Rads_2/561546/DBT/R_CC # Muita calc
                            (1392, 314, 31), # Bi-Rads_2/561546/DBT/R_MLO # Muita calc
                            (1129, 1087, 33), # Bi-Rads_2/574086/DBT/R_MLO
                            (1932, 1163, 7), # Bi-Rads_2/582336/DBT/L_CC
                            (1371, 550, 85), # Bi-Rads_2/582336/DBT/L_MLO
                            (1179, 717, 132), # Bi-Rads_2/603598/DBT/L_CC
                            (583, 1188, 110), # Bi-Rads_2/603598/DBT/L_MLO
                            (1233, 1490, 81), # Bi-Rads_2/603598/DBT/R_CC
                            (960, 1175, 87), # Bi-Rads_2/603598/DBT/R_MLO
                            (1687, 684, 45), # Bi-Rads_2/615333/DBT/R_CC
                            (1304, 386, 61), # Bi-Rads_2/615333/DBT/R_MLO
                            (0,0,0), # Bi-Rads_2/650301/DBT/R_MLO_1
                            (1207, 345, 47), # Bi-Rads_2/689606/DBT/R_CC
                          ]
                '''
            
            
            
            
        

            
            
            
            
                
                
