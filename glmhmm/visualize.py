#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:20:42 2021

@author: irisstone

Functions for visualizing and plotting results related to glmhmm fitting code
"""
import matplotlib.pyplot as plt
import numpy as np
from glmhmm.utils import find_best_fit
    
def plot_model_params(M,ax,precision='%.2f'):
    
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
                ax.text((j+1)/(J)-(1/(J+1)),((I-i)/I)-(1/(I+2)),precision %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)
            else:
                ax.text(0.3,((I-i)/I)-(1/(I+2)),precision %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)

def plot_loglikelihoods(lls,maxdiff,startix=5):
    '''
    Plot the trajectory of the log-likelihoods for multiple fits, identify how many top fits (nearly) match, and 
    color those trajectories in the plot accordingly
    '''
    
    # get the final ll for each fit
    final_lls = np.array([np.amax(lls[i,~np.isnan(lls[i,:])]) for i in range(lls.shape[0])])
    
    # get the index of the top ll
    bestix = find_best_fit(lls)
    
    # compute the difference between the top ll and all final lls
    ll_diffs = final_lls[bestix] - final_lls
    
    # identify te fits where the difference from the top ll is less than maxdiff
    top_matching_lls = lls[ll_diffs < maxdiff,:]
    
    # plot
    plt.plot(lls.T[startix:], color='black')
    plt.plot(top_matching_lls.T[startix:], color='red')
    plt.xlabel('iterations of EM', fontsize=16)
    plt.ylabel('log-likelihood', fontsize=16)
    
    return np.where(ll_diffs < maxdiff)[0] # return indices of best (matching) fits


def plot_weights(w,ax,xlabels=None,color=None,style='-',label=[''],switch=False, error=None):
    
    if switch:
        w = np.insert(w,3,w[:,0],axis=1)
        w = np.delete(w,0,axis=1)
    
    if color is not None:
        if error is not None:
            error = error[(w.shape[0])*(w.shape[0]-1):]
            error = np.reshape(error,(w.shape[0],w.shape[1]))
            for i in range(w.shape[0]):
                ax.errorbar(np.arange(w[i,:].shape[0]),w[i,:],yerr=error[i,:],fmt=style,color=color[i],label=label[i],linewidth=2)
        else:
            for i in range(w.shape[0]):
                ax.plot(w[i,:],style,color=color[i],label=label[i],linewidth=2)
    else:
        if error is not None:
            ax.errorbar(w.T,yerr=error,fmt=style,label=label)
        else:
            ax.plot(w.T,style,label=label)
    ax.set_ylabel('weight')
    if xlabels:
        ax.plot(xlabels,np.zeros((len(xlabels),1)),'k--')
        ax.set_xticks(np.arange(0,len(xlabels)))
        ax.set_xticklabels(xlabels,rotation=90)
        
    
    