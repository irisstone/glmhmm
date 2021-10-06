#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:46:44 2021

@author: irisstone

Miscellaneous helper functions for fitting and analyzing GLM/HMM code

"""

import numpy as np

def permute_states(A):
    
    'Permute states in order from highest to lowest self-transition value'
    
    diags = np.diagonal(A) # get diagonal values for sorting
    
    order = np.flip(np.argsort(diags))
    
    Aperm = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Aperm[i,j] = A[order[i],order[j]]
    
    return Aperm, order 

def find_best_fit(lls):

    return np.argmax(np.nanmax(lls,axis=1))
