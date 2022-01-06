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
from glmhmm.analysis import fit_line_to_hist
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# set plot design features
font = {'family'   : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight'   : 'regular',
        'size'     : 18}

mpl.rc('font', **font)
    
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

def plot_loglikelihoods(lls,maxdiff,ax,startix=5):
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
    ax.plot(np.arange(startix,lls.shape[1]),lls.T[startix:], color='black')
    ax.plot(top_matching_lls.T[startix:], color='red')
    ax.set_xlabel('iterations of EM', fontsize=16)
    ax.set_ylabel('log-likelihood', fontsize=16)
    
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
        
def plot_psychometrics(colors,title,file_path,save_path):

    import matlab.engine
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab')
    eng.addpath(s, nargout=0)

    # convert to matlab data types
    colors = matlab.double(colors)
    title = eng.convertCharsToStrings(title)
    file_path = eng.convertCharsToStrings(file_path)
    save_path = eng.convertCharsToStrings(save_path)

    ret = eng.fit_psychometrics(colors,title,file_path,save_path)

def plot_glmvsglmhmm_performance(data,label,color,avg_sess_length,ax,axis_len=80):
    ax.plot(np.arange(0,axis_len,0.001),np.arange(0,axis_len,0.001),'k--', linewidth=3)
    ax.plot(data[:,0]*avg_sess_length,data[:,1]*avg_sess_length,'o',markersize=10,color=color,label=label)

    ax.set_xlim([0.0,axis_len])
    ax.set_ylim([0.0,axis_len])
    ax.set_xticks(np.arange(10,75,20))
    ax.set_yticks(np.arange(10,75,20))
    ax.set_xticklabels(np.arange(10,75,20),fontsize=24)
    ax.set_yticklabels(np.arange(10,75,20),fontsize=24)
    ax.legend(fontsize=15, loc=4)

def plot_histogram_run_lengths(bin_heights,bin_edges,ax,color=[0,0,0],label=''):
    '''
    Recreates Fig 5E/F from the paper.

    Parameters
    ----------
    bin_heights : num_sims x num_bins array containing the value of each bin height of each histogram
    bin_edges : num_bins + 1 vector containing the values of the bin edges
    ax : the figure axis handle
    color : desired color for plotting, optional
    label : the label to be used in the legend, optional
    '''

    # determines whether to take average of multiple histograms based on shape of bin_heights
    if len(bin_heights.shape) > 1: 
        take_average = True
        num_bins = bin_heights.shape[1]
    else: 
        take_average = False
        num_bins = bin_heights.shape[0]

    if take_average:
        # compute statistics
        num_sims = bin_heights.shape[0]
        avg_bin_heights = np.mean(bin_heights,axis=1)
        std_bin_heights = np.std(bin_heights,axis=1)
        confidence_interval = avg_bin_heights + 1.96*(std_bin_heights/np.sqrt(num_sims))
        confidence_range = confidence_interval - avg_bin_heights
    else:
        avg_bin_heights = bin_heights

    # obtain smoothed curve from averaged bin heights
    smoothed_counts = fit_line_to_hist(avg_bin_heights,window_size=4)

    # plot results
    half_bin_width = (bin_edges[1]-bin_edges[0])/2
    x = np.linspace(bin_edges[0]-half_bin_width, bin_edges[-1]-half_bin_width, num_bins)
    ax.plot(x,smoothed_counts,color=color,label=label,linewidth=3)
    if take_average:
        ax.fill_between(x,smoothed_counts-confidence_range,smoothed_counts+confidence_range,color=color,alpha=0.3)
    ax.legend()
    ax.set_ylabel('counts')

    
    