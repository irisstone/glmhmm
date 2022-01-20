#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:46:44 2021

@author: irisstone

Various analysis functions for evaluating the results from fitting glm-hmms (as used in Bolkan, Stone et al 2022)

"""

import numpy as np
import scipy.io as sio
from glmhmm.utils import convert_ll_bits, reshape_obs, compObs
from glmhmm.glm import GLM
import matplotlib.pyplot as plt
import itertools

def compare_LL_GLMvsGLMHMM(fit_glm,fit_glmhmm,x,y):
    
    '''
    Computes the test loglikelihood from fitted parameters of GLM and GLM-HMM models 
    Parameters
    ----------
    fit_glm : fitted GLM object
    fit_glmhmm : fitted GLM-HMM object
    x : Nxm matrix of inputs
    y : Nx1 vector of observations
    
    Returns
    -------
    test_lls : a 2x1 vector containing the test lls (in bits) for the GLM (first entry) and GLM-HMM (second entry) 
    '''

    # create empty vector to store ll values
    test_lls = np.zeros((2))

    # compute L0 (LL of the bias-only GLM)
    glm_bias_only = GLM(fit_glm.n,1,fit_glm.c,observations="bernoulli")
    w_init = glm_bias_only.init_weights()
    w_bias, _ = glm_bias_only.fit(x[:,0,np.newaxis],w_init,y, compHess=False)
    L0 = glm_bias_only.ll

    # compute LL of GLM with all regressors
    y_reshaped = reshape_obs(y)
    LL_glm = -fit_glm.neglogli(x,fit_glm.w,y_reshaped)

    # compute LL of GLM-HMM with all regressors
    phi = np.zeros((len(y),fit_glmhmm.k,fit_glmhmm.c))
    for i in range(fit_glmhmm.k):
        phi[:,i,:] = compObs(x,fit_glmhmm.w[i])
    LL_glmhmm,_,_ = fit_glmhmm.forwardPass(y,fit_glmhmm.A,phi)

    # convert both to bits
    test_lls[0] = convert_ll_bits(LL_glm,L0,len(y))
    test_lls[1] = convert_ll_bits(LL_glmhmm,L0,len(y))
    
    return test_lls

def compare_predictions_GLMvsGLMHMM(fit_glm,fit_glmhmm,x,y,laser_only=False):
    '''
    Computes the prediction accuracy on test sets from fitted parameters of GLM and GLM-HMM models 
    Parameters
    ----------
    fit_glm : fitted GLM object
    fit_glmhmm : fitted GLM-HMM object
    x : Nxm matrix of inputs
    y : Nx1 vector of observations
    laser_only : boolean, optional -- if True, only computes prediction accuracy on subset of trials where laser is on. Default is False.
    
    Returns
    -------
    test_preds : a 2x1 vector containing the prediction accuracy on the tests sets for the GLM (first entry) and GLM-HMM (second entry) 
    '''

    if laser_only: 
        laserONix = x[:,2] != 0
        x = x[laserONix,:]
        y = y[laserONix]

    # create empty vector to store ll values
    test_preds = np.zeros((2))

    # compute prediction accuracy of GLM
    phi = fit_glm.compObs(x,fit_glm.w)
    test_preds[0] = np.sum(np.round(phi[:,1]) == y)/len(y)

    # compute prediction accuracy of GLM-HMM
    phi = np.zeros((len(y),fit_glmhmm.k,fit_glmhmm.c))
    for i in range(fit_glmhmm.k):
        phi[:,i,:] = compObs(x,fit_glmhmm.w[i])
    _,_,alpha_prior,_ = fit_glmhmm.forwardPass(y,fit_glmhmm.A,phi)
    pred_choice = np.zeros((len(y)))
    for i in range(len(y)):
        pred_prob = alpha_prior[i,:]@phi[i,:,1] # weighted observation probability
        pred_choice[i] = np.round(pred_prob) # predicted choice
    test_preds[1] = np.sum(pred_choice == y)/len(y)
    
    return test_preds

def fit_line_to_hist(bin_heights,window_size=3):
    '''
    Fits a line to a given list of histogram bin heights using a sliding window

    Parameters
    ----------
    bin_heights : histogram bin heights
    window_size : sliding window size

    Returns
    -------
    line : the fit line
    '''
    line = np.zeros(len(bin_heights),dtype=float)
    i = 0
    for bh in bin_heights:
        if i < 2:
            line[i] = bh
        elif i == len(bin_heights)-1:
            line[i] = 0
        else:
            line[i] = np.mean(bin_heights[i-1:i+3])
        i += 1
    return line

def blocks_of_laser_effect(sessions,y,laser,num_bins=40,bin_edges=None,min_run=2,side_bias=1):
    '''
    Identifies "runs" of consecutively identical choices in the same direction expected by inhibition during laser on trials

    Parameters
    ----------
    sessions : vector of starting indices of sessions in data (length equal to total number of sessions)
    y : vector of observations
    laser : vector of the status of laser-induced inhibition for each trial (length equal to total number of trials)
    num_bins : the number of bins to use when generating histograms of the run lengths
    min_run : number of consecutive identical choice, bias-consistent laser trials to qualify as a "run"
    side_bias : the direction of choice bias expected by the laser (1 = ipsi, -1 = contra)

    Returns
    -------
    bin_edges : the locations of the bin edges (can be used to align multiple histograms), vector of length num_bins
    bin_heights :  the values of the histogram bin heights for each simulation, matrix of size num_bins x simulations
    '''

    num_sessions = len(sessions)-1
    run_length_lists, laser_ixs_list = [],[]

    y[y==0]=-1 # change coding of choices from 0/1 to -1/1

    for j in range(num_sessions):
        # get values for individual session
        y_sess = y[sessions[j]:sessions[j+1]]
        laser_sess = laser[sessions[j]:sessions[j+1]]

        # get choices only for laser on trials
        laserON_ixs = np.where(laser_sess != 0)[0]
        laserON_laser_session = laser_sess[laserON_ixs]
        y_laser_session = y_sess[laserON_ixs]

        # we only care about runs where the choice is consistent with the bias effect we expect from the laser
        y_laser_bias = y_laser_session * laserON_laser_session # elems=1 if choice is in expected bias direction
        # now collect run lengths that are >min_run_length
        run_length = 1
        run_lengths = []

        for i in range(1,len(y_laser_bias)):
            if y_laser_bias[i] == y_laser_bias[i-1] == side_bias: # if the same as previous laser choice                                 ,,,,,
                run_length += 1

            else: # change in laser effect
                if run_length >= min_run: # if meets conditions, store run
                    run_lengths.append(run_length)
                    laser_ixs_list.append(list(laserON_ixs[i+1-run_length:i+1])) # store indices of trials in each run

                run_length = 1 # after storing run or if run not long enough, reset run length
        if run_length >= min_run: 
                run_lengths.append(run_length)
                laser_ixs_list.append(list(laserON_ixs[i+1-run_length:i+1]))

        run_length = 1
        run_length_lists.append(run_lengths)

    all_run_lengths = np.array(list(itertools.chain(*run_length_lists))) # flatten list of lists into one list

    # get length of run from first laser trial to last (add one to include both edges of list)
    run_lengths_all_trials = [sublist[-1] - sublist[0] + 1 for sublist in laser_ixs_list] 

    if bin_edges is None:
        bh,bin_edges,patches = plt.hist(all_run_lengths,align='mid',bins=num_bins,label='data',lw=2,color='k')

    bin_heights,_,_  = plt.hist(all_run_lengths,align='mid',bins=bin_edges)
    plt.close()

    return bin_edges, bin_heights

def session_lengths_for_animal(animal_IDs,unique_animal_IDs,session_IDs):

    ixs = np.where(animal_IDs == unique_animal_IDs)[0] # get trial ixs for one mouse at a time
    session_IDs = session_IDs[ixs] # get session IDs for that mouse
    _, session_lengths = np.unique(session_IDs,return_counts = True)

    return ixs,session_lengths

def dwell_times_per_session(z,dwell_times=None,terminal_run=False):
    '''
    Gets the dwell times associated with each state in a particular session
    Parameters
    ----------
    z : the state probabilities associated with a particular session
    dwell times : optional, a list of existing dwell times to append to
    terminal run : boolean, optional, determines whether to include the last run in the session in 
    computing dwell times. Default False (as the terminal runs are truncated prematurely by the end
    of the session)
    '''

    if dwell_times is None:
        K = len(np.unique(z))
        dwell_times = [[] for i in range(K)] # initialize empty list for each state

    # loop through each trial for each session
    run_length = 1 # start run length at one
    state = int(z[0]) # get state assignment for initial run
    for k in range(1,len(z)):  
        if z[k] == z[k-1]:
            run_length += 1 # add to run length
        else:
            dwell_times[state].append(run_length) # append run length to appropriate list
            state = int(z[k]) # get state assignment for next run
            run_length = 1 # reset run length

    # include last run length of session
    if terminal_run:
        dwell_times[state].append(run_length) 

    return dwell_times
