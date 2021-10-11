#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:46:44 2021

@author: irisstone

Miscellaneous helper functions for fitting and analyzing GLM/HMM code

"""

import numpy as np

def permute_states(M,method='self-transitions',param='transitions',order=None):
    
    '''
    Parameters
    ----------
    M : matrix of probabilities for input parameter (transitions, observations, or initial states)
    Methods --- 
        self-transitions : permute states in order from highest to lowest self-transition value (works
             only with transition probabilities as inputs)
        order : permute states according to a given order
    param : specifies the input parameter
    order : optional, specifies the order of permuted states for method=order
    
    Returns
    -------
    M_perm : M permuted according to the specified method/order
    order : the order of the permuted states
    '''
    
    # check for valid method
    method_list = {'self-transitions','order'}
    if method not in method_list:
        raise Exception("Invalid method: {}. Must be one of {}".
            format(method, method_list))
        
    # sort according to transitions
    if method =='self-transitions':
        
        if param != 'transitions':
            raise Exception("Invalid parameter choice: self-transitions permutation method \
                            requires transition probabilities as parameter function input")
        diags = np.diagonal(M) # get diagonal values for sorting
        
        order = np.flip(np.argsort(diags))
        
        M_perm = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_perm[i,j] = M[order[i],order[j]]
                
    # sort according to given order
    if method == 'order':
        if param=='transitions':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M_perm[i,j] = M[order[i],order[j]]
        if param=='observations':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                M_perm[i,:] = M[order[i],:]
    
    return M_perm, order 

def find_best_fit(lls):

    return np.argmax(np.nanmax(lls,axis=1))
