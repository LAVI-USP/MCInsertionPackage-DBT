#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:56:42 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ref: https://stackoverflow.com/a/65873479/8682939

def binomial_ci(x, n, alpha=0.05):
    # x is number of successes, n is number of trials
    
    if x==0:
        c1 = 0
    else:
        c1 = stats.beta.interval(1-alpha, x,n-x+1)[0]
    if x==n:
        c2=1
    else:
        c2 = stats.beta.interval(1-alpha, x+1,n-x)[1]
        
    return c1, c2


if __name__ == '__main__':
    
    readers = [47, 42, 56, 41, 67, 70]
    
    number_of_trials = 100 * len(readers)
    
    c1_all, c2_all = binomial_ci(number_of_trials / 2, number_of_trials, alpha=0.05)
    
    print(c1_all, c2_all)
    
    print(np.mean(readers))
    
    c1, c2 = binomial_ci(50, 100, alpha=0.05)

    
    fig, ax1 = plt.subplots()    
    plt.bar(np.arange(len(readers)) + 1, 
             readers, 
             facecolor=(0.26953125, 0.44921875, 0.765625), 
             edgecolor='k',
             alpha=0.6, 
             label='Reader selection rate')
    
    ax1.fill_between(np.array([0.5, 1, 2, 3, 4, 5, 6.5]), c1*100, c2*100, 
                      color='gray', 
                      alpha=.4,
                      label='95% confidence interval of a random choice')
    
    
    # ax1.fill_between(np.array([0.5, 1, 2, 3, 4, 5, 6.5]), c1_all*100, c2_all*100, 
    #                   color='black', 
    #                   edgecolor='black',
    #                   alpha=.2,
    #                   label='95% confidence interval of study')
    
    # plt.plot(np.arange(0,10), np.mean(readers) * np.ones((10)), 
    #          'k--',
    #          label='Readers mean')
    
    plt.plot(np.arange(0,10), 50 * np.ones((10)), 
              'k--')
    
    
    ax1.set_xlabel("Readers", fontsize=16)
    ax1.set_ylabel("Successes (%)", fontsize=16)
    
    ax1.set_xticks(np.arange(len(readers)) + 1)
    
    ax1.set_xlim(0.41, 6.59)
    ax1.set_ylim(0, 80)
    
    ax1.legend()
    
    fig.tight_layout()
    
    fig.savefig("2afc_result.png")