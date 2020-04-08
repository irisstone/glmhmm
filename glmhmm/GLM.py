#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:06:06 2019

@author: istone

Class for fitting generalized linear models (GLMs).

Updated: April 8th, 2020

"""
#import numpy as np
import autograd.numpy as np
from scipy import optimize
from autograd import value_and_grad, hessian

class GLM(object):
    
    """
    Base class for fitting generalized linear models. 
    Notation: 
        n: number of data points
        m: number of features
        c: number of classes 
        x: design matrix (nxm)
        y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)

    """
    def __init__(self,n,m,c):
        self.n, self.m, self.c = n, m, c
        
        
    def neglogli(self,x,w,y):
        """
        For known values of x, w, and y, calculate the total loglikelihood p(y|x)
        """
        
        try:
            w.shape[1]
        except IndexError:
            w = w[:,np.newaxis]
            
        p = np.exp(x@w) # get exponentials e^(wTx)
        p = np.hstack((p,np.ones((len(p),1))))
        pT1 = np.sum(p,axis=1) # get normalization constant (sum of exponentials) --> pT1 ("one hot representation")
        pTy = np.sum(np.multiply(p,y),axis=1)
        log_pyx = np.log(pTy) - np.log(pT1)
        
        assert np.round(np.sum(np.divide(np.exp(x@w).T,np.sum(np.exp(x@w)))),2) == 1, 'Sum of normalized probabilities does not equal 1!'
        
        self.ll = -np.sum(log_pyx)
        self.x, self.w, self.y = x, w, y
        
        return -np.sum(log_pyx)
    
    def fit(self,x,w,y,compHess = False):
        """
        Use gradient descent to optimize weights
        """
        
        # optimize loglikelihood given weights
        w_flat = np.ndarray.flatten(w[:,0:-1]) # starting weights for optimization    
        opt_log = lambda w: self.neglogli(x,w,y) # calculate log likelihood 
        OptimizeResult = optimize.minimize(value_and_grad(opt_log),w_flat, jac = "True", method = "L-BFGS-B")
       
        w_updated = np.hstack((np.reshape(OptimizeResult.x,(self.m,self.c-1)),np.zeros((self.m,1))))
        
        # Get updated probabilities (these become observation probabilities in HMM)
        probs_out = np.exp(np.dot(x,w_updated)) # get exponentials e^wTx
        pT1 = np.sum(probs_out,axis=1) # get normalization constant (sum of exponentials) --> pT1 ("one hot representation")
        obs = np.divide(probs_out.T,pT1).T # final probabilities calculated with updated weights
        
        
        if compHess:
            ## compute Hessian
            hess = hessian(opt_log) # function that computes the hessian
            H = hess(w_updated[:,:-1]) # gets matrix for w_hats
            variance = np.sqrt(np.diag(np.linalg.inv(H.T.reshape((self.m * (self.c-1),self.m * (self.c-1)))))) # calculate variance of weights from Hessian
        else: 
            variance = []
        
        self.w, self.obs, self.variance = w_updated, obs, variance
        
        return obs, w_updated