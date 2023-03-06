#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:56:42 2022

@author: Rodrigo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KVP = list(df.nVoxels)
unique_kvp = np.unique(KVP)
fig, ax1 = plt.subplots()    
ax1.hist(KVP, 
         3,#np.linspace(unique_kvp[0]-0.5, unique_kvp[-1]+0.5, unique_kvp.shape[0]+1), 
         rwidth=0.8, 
         facecolor=(0.0, 0.6875, 0.30859375), 
         edgecolor='k',
         alpha=0.8)
ax1.set_xlabel("#Voxels", fontsize=16)
ax1.set_ylabel("Occurrences", fontsize=16)
# ax1.set_xticks(unique_kvp)
fig.tight_layout()
fig.savefig("nVoxels.png")
