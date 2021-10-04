#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:20:42 2021

@author: irisstone

Functions for visualizing and plotting results related to glmhmm fitting codd
"""
import matplotlib.pyplot as plt
import numpy as np

    
def plot_transitions(A):
    
    # plot heat map of transitions
    fig, ax = plt.subplots()
    plt.imshow(A,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    # add numerical values to plot
    k = A.shape[0]
    for i in range(k):
        for j in range(k):
            if i == j:
                color='black'
            else:
                color='white'

            plt.text((j+1)/(k)-(1/(k+1)),((k-i)/k)-(1/(k+2)),'%.2f' %(A[i,j]),transform=ax.transAxes,fontsize=15,color=color)
