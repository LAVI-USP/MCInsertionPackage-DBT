#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:38:18 2022

@author: rodrigo

To remove:

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
    
    coords = [  (1182, 1689, 80), # Bi-Rads_2/2206283/DBT/L_CC # Crop 362
                (1043, 1619, 71), # Bi-Rads_2/2206283/DBT/L_MLO # Crop 362
                (1294, 997, 147), # Bi-Rads_2/1063739/DBT/R_MLO
                (1312, 602, 72), # Bi-Rads_2/1080674/DBT/L_CC
                (1594, 451, 47), # Bi-Rads_2/1080674/DBT/L_MLO
                (1222, 410, 78), # Bi-Rads_2/1080674/DBT/R_CC
                (1282, 378, 65), # Bi-Rads_2/1080674/DBT/R_MLO
                (1198, 1741, 36), # Bi-Rads_2/1474181/DBT/R_CC
                (1108, 596, 51), # Bi-Rads_2/1474181/DBT/R_MLO
                (0, 0, 0), # Bi-Rads_2/1491225/DBT/L_MLO
                (1118, 497, 66), # Bi-Rads_2/1682942/DBT/R_CC
                (1554, 514, 88), # Bi-Rads_2/1682942/DBT/R_MLO
                (1190, 1832, 9), # Bi-Rads_2/1694809/DBT/L_CC
                (1145, 319, 16), # Bi-Rads_2/1694809/DBT/L_MLO_1
                (1546, 375, 43), # Bi-Rads_2/1723318/DBT/R_MLO
                (1157, 664, 64), # Bi-Rads_2/1739141/DBT/R_CC
                (677, 802, 44), # Bi-Rads_2/1739141/DBT/R_MLO_2
                (1110, 874, 129), # Bi-Rads_2/1741806/DBT/R_CC_1
                (1278, 837, 68), # Bi-Rads_2/1741806/DBT/R_MLO
                (1199, 1629, 15), # Bi-Rads_2/1760635/DBT/L_CC
                (1148, 1777, 10), # Bi-Rads_2/1760635/DBT/L_CC
                (1189, 1714, 13), # Bi-Rads_2/1760635/DBT/R_MLO_2
                (1339, 1782, 31), # Bi-Rads_2/1778779/DBT/R_CC
                (1258, 568, 66), # Bi-Rads_2/1778779/DBT/R_MLO_1
                (1331, 584, 139), # Bi-Rads_2/1808158/DBT/L_MLO
                (1732, 568, 25), # Bi-Rads_2/1808158/DBT/R_CC
                (1586, 538, 25), # Bi-Rads_2/1808158/DBT/R_MLO
                (1278, 1622, 66), # Bi-Rads_2/1870439/DBT/L_MLO
                (1221, 323, 108), # Bi-Rads_2/1907565/DBT/R_MLO
                (1274, 1740, 54), # Bi-Rads_2/2038867/DBT/R_CC
                (1233, 758, 72), # Bi-Rads_2/2038867/DBT/R_MLO
                (796, 818, 60), #  Bi-Rads_2/2060247/DBT/L_CC_2
                (934, 914, 52), # Bi-Rads_2/2060247/DBT/L_MLO
                (1245, 1179, 71), # Bi-Rads_2/2162013/DBT/L_MLO
                (928, 755, 55), # Bi-Rads_2/2165900/DBT/L_CC
                (1257, 627, 62), # Bi-Rads_2/2165900/DBT/L_MLO
                (1097, 1519, 89), # Bi-Rads_2/2178692/DBT/L_MLO
                (926, 1829, 69), # Bi-Rads_2/2195734/DBT/L_CC
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
                (1796, 374, 81), # Bi-Rads_2/2375168/DBT/L_CC
                (1348, 578, 71), # Bi-Rads_2/2375168/DBT/L_MLO
                (1230, 1786, 54), # Bi-Rads_2/2435574/DBT/L_CC_2
                (1205, 453, 59), # Bi-Rads_2/2435574/DBT/L_MLO
                (1273, 660, 81), # Bi-Rads_2/2463088/DBT/L_MLO
                (1556, 1825, 21), # Bi-Rads_2/2463187/DBT/R_CC
                (0,0,0),
                (1637, 665, 43), # Bi-Rads_2/2482750/DBT/L_CC
                (1509, 626, 60), # Bi-Rads_2/2482750/DBT/L_MLO
                (746, 538, 64), # Bi-Rads_2/2489465/DBT/L_CC
                (766, 440, 55), # Bi-Rads_2/2489465/DBT/L_MLO_1
                (1557, 376, 132), # Bi-Rads_2/2489465/DBT/R_CC
                (1097, 399, 134), # Bi-Rads_2/2489465/DBT/R_MLO_1
                (1332, 556, 37), # Bi-Rads_2/2548326/DBT/L_CC # Muita calc
                (1352, 474, 67), # Bi-Rads_2/2548326/DBT/L_MLO # Muita calc
                (1156, 798, 111), # Bi-Rads_2/2566045/DBT/L_CC
                (609, 554, 113), # Bi-Rads_2/2566045/DBT/L_MLO
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
                (1419, 1042, 50), # Bi-Rads_2/689606/DBT/R_MLO
                (1135, 386, 59), # Bi-Rads_2/816262/DBT/R_CC
                (1080, 363, 51), # Bi-Rads_2/816262/DBT/R_MLO
                (1500, 327, 50), # Bi-Rads_3/596877/DBT/L_CC
                (1446, 325, 53), # Bi-Rads_3/596877/DBT/L_MLO
                (1484, 497, 43), # Bi-Rads_3/596877/DBT/R_CC
                (1411, 623, 36), # Bi-Rads_3/596877/DBT/R_MLO
                (1215, 1788, 52), # 113006/raw/DBT/LCC
                (546, 1577, 43), # 113006/raw/DBT/LMLO
                (0,0,0), # 1557516/raw/DBT/RCC
                (0,0,0), # 1557516/raw/DBT/RMLO
                (0,0,0), # 1568677/raw/DBT/RCC
                (900, 1544, 30), # 1568677/raw/DBT/RMLO
                (0,0,0), # 1570683/raw/DBT/LCC
                (0,0,0), # 1570683/raw/DBT/LMLO
                (1390, 315, 71), # 2055850/raw/DBT/RCC
                (1447, 312, 84), # 2055850/raw/DBT/RMLO
                (0,0,0), # 2169181/raw/DBT/RCC
                (0,0,0), # 2169181/raw/DBT/RMLO
                (1713, 829, 96), # 2484839/raw/DBT/RCC
                (1455, 1159, 37), # 2537185/raw/DBT/RCC
                (1593, 1096, 48), # 2537185/raw/DBT/RMLO
                (1213, 789, 32), # 2589480/raw/DBT/LCC
                (0, 0, 0), # 2589480/raw/DBT/LMLO
                (1269, 583, 41), # 3122776/raw/DBT/RMLO
                (0, 0, 0), # 450852/raw/DBT/RCC
                (1239, 569, 32), # 629180/raw/DBT/LCC
                (1750, 691, 54), # 629180/raw/DBT/LMLO
                (1172, 424, 63), # 629180/raw/DBT/RCC
                (1114, 343, 48), # 629180/raw/DBT/RMLO
              ]
    
    ignore_cases = ['Bi-Rads_2/1491225/DBT/L_MLO',
        'Bi-Rads_2/2265387/DBT/R_CC',
        'Bi-Rads_2/2265387/DBT/R_MLO',
        'Bi-Rads_2/2722805/DBT/L_CC_2',
        'Bi-Rads_2/2722805/DBT/L_MLO',
        'Bi-Rads_2/650301/DBT/R_MLO_1',
        'Bi-Rads_2/2463187/DBT/R_MLO',
        'Bi-Rads_2/486871/DBT/R_CC',
        'Bi-Rads_2/486871/DBT/R_MLO',
        'Bi-Rads_2/1080674/DBT/R_MLO', 
        'Bi-Rads_2/1080674/DBT/R_CC', 
        'Bi-Rads_2/689606/DBT/R_MLO', 
        'Bi-Rads_2/615333/DBT/R_CC', 
        'Bi-Rads_2/2060247/DBT/L_CC_2', 
        'Bi-Rads_2/2060247/DBT/L_MLO', 
        'Bi-Rads_2/2324391/DBT/L_MLO', 
        'Bi-Rads_2/2646039/DBT/R_MLO', 
        'Bi-Rads_2/1741806/DBT/R_MLO', 
        'Bi-Rads_2/2463088/DBT/L_MLO', 
        'Bi-Rads_2/478677/DBT/L_MLO', 
        'Bi-Rads_2/2217424/DBT/L_CC', 
        'Bi-Rads_2/2217424/DBT/L_MLO_1', 
        'Bi-Rads_2/1739141/DBT/R_MLO_2', 
        'Bi-Rads_2/486871/DBT/L_CC',
        'Bi-Rads_2/603598/DBT/R_MLO', 
        'Bi-Rads_2/603598/DBT/R_CC', 
        'Bi-Rads_2/561546/DBT/R_MLO', 
        'Bi-Rads_2/486871/DBT/L_CC', 
        'Bi-Rads_2/561546/DBT/L_CC', 
        'Bi-Rads_2/2178692/DBT/L_MLO',
        '1557516/raw/DBT/RCC',
        '1557516/raw/DBT/RMLO',
        '1568677/raw/DBT/RCC',
        '1570683/raw/DBT/LCC',
        '1570683/raw/DBT/LMLO',
        '2169181/raw/DBT/RCC',
        '2169181/raw/DBT/RMLO',
        '2589480/raw/DBT/LMLO',
        '450852/raw/DBT/RCC'
        ]
    
    recon_dirs = [str(item) for item in pathlib.Path(path2read).glob("*/**") if item.is_dir() and ('CC' in str(item) or 'MLO' in str(item))]
    
    # coord = (0,0)

    # for idX, (recon_dir, coord) in enumerate(zip(recon_dirs, coords)):
    for idX, recon_dir in enumerate(recon_dirs):
        
        print('Processing case: {}'.format("/".join(recon_dir.split('/')[-4:])))
        
        path2write_patient_name = path2write + "/" + "/".join(recon_dir.split('/')[-4:])
        
        if "/".join(recon_dir.split('/')[-4:]) in ignore_cases:
            continue
        
        if not makedir(path2write_patient_name):
            
            vol = readDicom(recon_dir)
            
            # a = 1
            
            coord = coords[idX] #(1114, 343, 48)
            
            # plt.imshow(vol[coord[0]:coord[0]+200, coord[1]:coord[1]+200, coord[2]], 'gray')
                        
            for z in range(-7,8):
                
                dcmFile_tmp = path2write_patient_name + '{}{}.dcm'.format(filesep(), coord[2] + z)
                
                writeDicom(dcmFile_tmp, np.uint16(vol[coord[0]:coord[0]+200, coord[1]:coord[1]+200, coord[2] + z]))
                
            
            
            
            
        
                
