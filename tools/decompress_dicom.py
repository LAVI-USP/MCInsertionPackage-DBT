#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:24:27 2022

@author: rodrigo
"""


import pathlib
import pydicom
import sys

from tqdm import tqdm

sys.path.insert(1, '../')

from libs.utilities import makedir, filesep, writeDecompressedDicom


if __name__ == '__main__':
    
    path2read = '/media/rodrigo/Dados_2TB/Imagens/HC_Barretos/Imagens_Clinicas_Pristina_Out_2022/raws_02'
    
    dcm_files = [str(item) for item in pathlib.Path(path2read).glob("**/*.dcm")]
    
    for dcm_file in tqdm(dcm_files):
        
        proj_header = pydicom.dcmread(dcm_file, force=True)
        proj  = proj_header.pixel_array
        
        writeDecompressedDicom(dcm_file, proj, proj_header)
