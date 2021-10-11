#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:20:42 2021

@author: irisstone

Functions for visualizing and plotting results related to glmhmm fitting codd
"""
import matplotlib.pyplot as plt
import numpy as np

    
def plot_model_params(M,ax):
    
    # plot heat map of transitions
    ax.imshow(M,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # add numerical values to plot
    k = M.shape[0]
    for i in range(k):
        for j in range(k):
            if M[i,j] >= 0.5:
                color='black'
            else:
                color='white'

            ax.text((j+1)/(k)-(1/(k+1)),((k-i)/k)-(1/(k+2)),'%.2f' %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)
