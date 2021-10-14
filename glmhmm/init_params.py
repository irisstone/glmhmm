#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:11:13 2020

@author: istone

Functions for initializing the parameters of a hidden Markov Model (\theta = {A,\phi,\pi}) or GLM-HMM (\theta = {A,w,\pi})
using different distributions. Can easily be extended to include support for custom distributions. 
"""
import numpy as np
#from glmhmm import glm

def init_transitions(self,distribution='dirichlet',alpha_diag=5,alpha_full=1):
    
    """
    Initializes values for the transition probabilities.

    Parameters
    ----------
    distribution : string, optional
        Sets the distribution to use when initializing the transition probabilities. The default is dirichlet.
    alpha_diag : int, optional
        Sets the concentration parameter for the diagonal values when using a Dirichlet distribution. Default is 5. 
    alpha_full : int, optional
        Sets the concentration parameter for the off-diagonal values when using a Dirichlet distribution. Default is 1.

    Returns
    -------
    A : kxk matrix of the transition probabilities

    """
    
    if distribution == 'dirichlet':
    
        # Make transition matrix by sampling each row from Dirichlet distribution (achieved by normalizing gamma random variables)
        A = np.random.gamma(alpha_full*np.ones((self.k,self.k)) + alpha_diag*np.identity(self.k),1)
        A = A/np.repeat(np.reshape(np.sum(A,axis=1),(1,self.k)),self.k,0).T
        
    elif distribution == 'uniform':
        
        # Make transition matrix probabilities uniform 
        A = (1/self.k) * np.ones((self.k,self.k))
    
    return A

def init_states(self,distribution='uniform'):
    
    """
    Initializes values for the state probabilities at t=1.

    Parameters
    ----------
    distribution : string, optional
        Sets the distribution to use when initializing the state probabilities for t=1. The default is uniform (same probability for each state).

    Returns
    -------
    A : kx1 vector of the state probabilities for t=1

    """
    
    if distribution == 'uniform':
        pi = (1/self.k) * np.ones((self.k,1))
        
    if distribution == 'normal':
        pi = np.random.normal(loc=2,size=(self.k,1))
        pi = pi/np.sum(pi) # normalize so values add up to 1
        if np.any(pi) < 0:
            pi = pi + abs(np.min(pi)) # threshold so smallest value is > 0
    
    return pi

def init_emissions(self,distribution='dirichlet',alpha_diag=5,alpha_full=1):
    
    """
    Initializes values for the emission (observation) probabilities.

    Parameters
    ----------
    distribution : string, optional
        Sets the distribution to use when initializing the emission probabilities. The default is dirichlet.
    alpha_diag : int, optional
        Sets the concentration parameter for the diagonal values when using a Dirichlet distribution. Default is 5. 
    alpha_full : int, optional
        Sets the concentration parameter for the off-diagonal values when using a Dirichlet distribution. Default is 1.

    Returns
    -------
    phi : nxc matrix of the emission probabilities

    """
    
    if distribution == 'dirichlet':
        # Make emissions matrix by sampling each row from Dirichlet distribution (achieved by normalizing gamma random variables)
        phi = np.random.gamma(alpha_full,1,(self.k,self.c))
        if self.c > self.k: 
            phi = phi + np.append(alpha_diag*np.identity(self.k), np.zeros((self.k,self.c-self.k)),axis=1) # add to diagonal
        elif self.k > self.c: 
            phi = phi + np.append(alpha_diag*np.identity(self.c), np.zeros((self.k-self.c,self.c)),axis=0) # add to diagonal
        elif self.k == self.c:
            pass

        phi = phi/(np.repeat(np.reshape(np.sum(phi,axis=1),(1,self.k)),self.c,0).T) # normalize so columns sum to 1
        
    elif distribution == 'uniform':
        
        # Make emission matrix probabilities uniform 
        phi = (1/self.c) * np.ones((self.k,self.c))
    
    return phi

def init_weights(self,distribution='uniform',params=None,bias=True):
    """
    Initializes values for the weights 

    Parameters
    ----------
    distribution : string, the distribution from which to generate weights. The default is 'uniform'.
    params : list, any additional parameters required for generating the weights, e.g. the low and high bounds 
    for a uniform distribution or the mean and standard deviation for a normal distribution. 
    bias : boolean, specifies whether to add an offset to the weights. The default is True.

    Returns
    -------
    w : dxc matrix of weights

    """
    
    if distribution == 'uniform':
        
        try: self.k
        except NameError:
            # if number of states doesn't exist, initialize dxc weights
            w = np.random.uniform(params[0],high=params[1],size=(self.d,self.c-1))
            self.w = np.hstack((np.zeros((self.d,1)),w)) # add vector of zeros to weights
        else:
            # if number of states does exist, initialize kxdxc weights
            w = np.round(np.random.uniform(params[0],high=params[1],size=(self.k,self.d,self.c-1)),2)
            self.w = np.concatenate((np.zeros((self.k,self.d,1)),w),axis=2) # add vector of zeros to weights
        
    elif distribution == 'normal':
        try: self.k
        except NameError:
            w = np.random.normal(loc=params[0],scale=params[1],size=(self.d,self.c-1))
            self.w = np.hstack((np.zeros((self.d,1)),w)) # add vector of zeros to weights
        else:
            w = np.random.normal(loc=params[0],scale=params[1],size=(self.k,self.d,self.c-1))
            self.w = np.concatenate((np.zeros((self.k,self.d,1)),w),axis=2) # add vector of zeros to weights
        
    elif distribution == 'GLM':
        
        w = np.random.uniform(params[0],high=params[1],size=(self.d,self.c-1))
        self.w = np.hstack((np.zeros((self.d,1)),w)) # add vector of zeros to weights
        
        # reshape y from vector of indices to one-hot encoded array for matrix operations in glm.fit
        yint = params[3].astype(int)
        yy = np.zeros((yint.shape[0], yint.max()+1))
        yy[np.arange(yint.shape[0]),yint] = 1
        
        w, phi = self.glm.fit(params[2],self.w,yy,compHess=False,gammas=None,gaussianPrior=0)
        
        wk = np.zeros((self.k,self.d,self.c))
        for zi in range(self.k):
            noise = np.random.normal(loc=0,scale=1,size=(self.d,self.c-1)) # create vector of noise to add to weights
            noise = np.hstack((np.zeros_like(noise),noise)) # add vector of zeros to last column of weights
            wk[zi,:,:] = w + noise # add noise to weights 
            
        wk[:,0,1] = 1 # add uniform bias weight
            
        self.w = wk
        
        
    if bias:
        
        w[0,1:] = 1 # add bias to all except first column (which should stay all zeros)
        
    
    return self.w