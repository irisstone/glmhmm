#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:46:44 2021

@author: irisstone

Various analysis functions for evaluating the results from fitting glm-hmms (as used in Bolkan, Stone et al 2022)

"""

import numpy as np
import scipy.io as sio
from glmhmm.utils import convert_ll_bits

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
    L0 = -fit_glm.neglogli(x[:,0],fit_glm.w,y)

    # compute LL of GLM with all regressors
    LL_glm = -fit_glm.neglogli(x,fit_glm.w,y)

    # compute LL of GLM-HMM with all regressors
    LL_glmhmm,_,_ = fit_glmhmm.forwardpass(y,fit_glmhmm.A,fit_glmhmm.phi)

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
        y = y[laserONix,:]

    # create empty vector to store ll values
    test_preds = np.zeros((2))

    # compute prediction accuracy of GLM
    phi = fit_glm.compObs(x,fit_glm.w)
    test_preds[0] = np.sum(np.round(phi[:,1]) == y)/len(y)

    # compute prediction accuracy of GLM-HMM
    phi = np.zeros((len(y),fit_glmhmm.K,fit_glmhmm.C))
    for i in range(fit_glmhmm.K):
        phi[:,i,:] = fit_glm.compObs(x,fit_glm.w)
    _,alpha,_ = fit_glmhmm.forwardpass(y,fit_glmhmm.A,phi)
    pred_choice = np.zeros((len(y)))
    for i in range(len(y)):
        pred_prob = alpha[i,:]@phi[i,:,1] # weighted observation probability
        pred_choice[i] = np.round(pred_prob) # predicted choice
    test_preds[1] = np.sum(pred_choice == y)/len(y)
    
    return test_preds



