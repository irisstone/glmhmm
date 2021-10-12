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
    I = M.shape[0]
    J = M.shape[1]
    for i in range(I):
        for j in range(J):
            if M[i,j] >= 0.5:
                color='black'
            else:
                color='white'

            if J > 1:
                ax.text((j+1)/(J)-(1/(J+1)),((I-i)/I)-(1/(I+2)),'%.2f' %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)
            else:
                ax.text(0.3,((I-i)/I)-(1/(I+2)),'%.2f' %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)
