#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:46:44 2021

@author: irisstone

Miscellaneous helper functions for fitting and analyzing GLM/HMM code

"""

import numpy as np

def permute_states(M,method='self-transitions',param='transitions',order=None,ix=None):
    
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
    method_list = {'self-transitions','order','weight value'}
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
        if param=='weights':
            M_perm = np.zeros_like(M)
            for i in range(M.shape[0]):
                M_perm[i,:,:] = M[order[i],:,:]
                
    # sort by the value of a particular weight
    if method == 'weight value':
        if ix is None:
            raise Exception("Index of weight ix must be specified for this method")
        
        order = np.flip(np.argsort(M[:,ix]))
        
        M_perm = np.zeros_like(M)
        for i in range(M.shape[0]):
            M_perm[i,:] = M[order[i],:]
    
    return M_perm, order 


def find_best_fit(lls):

    return np.argmax(np.nanmax(lls,axis=1))

def compare_top_weights(w,ixs,tol=0.05):
    '''
    compares the weights associated with the top lls from multiple glm-hmm fits and checks if each 
    weight matches within the given tolerance
    '''
    
    best_weights = w[ixs[0],:,:]
    diff = np.zeros((len(ixs),w.shape[1],w.shape[2]))
    for i in range(1,len(ixs)): # for each specified fit (associated with the top lls)
        for j in range(w.shape[1]): # for each state
            diff[i-1,j,:] = abs(best_weights[j,:] - w[ixs[i],j,:])
            
    if np.any(diff > tol):
        print('One or more weights differ by more than the set tolerance. The largest difference was %.2f.' %(np.max(diff)))
        print('Try changing the tolerance or decreasing the number of top fits to compare.')
    else:
        print('None of the weights differ by more than the set tolerance. The largest difference was %.2f.' %(np.max(diff)))
        print('This confirms that the top fits (as specified) all converged on the same solution.')
