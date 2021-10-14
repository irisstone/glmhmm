#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:38:30 2020

@author: istone

Defines different distribution options for GLM observations.
"""

import autograd.numpy as np
#import jax.numpy as jnp

class Observations(object):
    
    def __init__(self,n,d,c):
       self.n, self.d, self.c = n, d, c
        
class BernoulliObservations(object):
    
    def __init__(self,n,d,c):
        self.n, self.d, self.c = n, d, c
    
    def compObs(self,x,w,normalize=True):
        """
        Computes the GLM observation probabilities for each data point.

        Parameters
        ----------
        x : nxm array of the data (design matrix)
        w : mxc array of weights
        normalize : boolean, optional
            Determines whether or not observation probabilities are normalized. The default is True.

        Returns
        -------
        phi : nxc array of the observation probabilities

        """
        
        assert self.c == 2, "A Bernoulli distribution must only have two observation classes (c=2)"
        
        phi = np.exp(x@w) # get exponentials e^(wTx)
        if normalize:
            try:
                phi = np.divide(phi.T,np.sum(phi,axis=1)).T # normalize the exponentials 
            except:
                phi = np.divide(phi.T,np.sum(phi)).T # normalize the exponentials 
        
        return phi
    
class MultinomialObservations(object):
    
    def __init__(self,n,d,c):
        self.n, self.d, self.c = n, d, c
    
    def compObs(self,x,w,normalize=True):
        """
        Computes the GLM observation probabilities for each data point.

        Parameters
        ----------
        x : nxm array of the data (design matrix)
        w : mxc array of weights
        normalize : boolean, optional
            Determines whether or not observation probabilities are normalized. The default is True.

        Returns
        -------
        phi : nxc array of the observation probabilities

        """
        
        assert self.c > 2, "A multinomial distribution should have more than two observation classes (c>2)"
        
        phi = np.exp(x@w) # get exponentials e^(wTx)
        if normalize:
            if len(phi.shape) == 2:
                phi = np.divide(phi.T,np.sum(phi,axis=1)).T # normalize the exponentials 
                assert np.all(np.round(np.sum(phi,axis=1),5)) == 1, "emission probabilities don't sum to 1!"
            if len(phi.shape) == 1:
                phi = np.divide(phi.T,np.sum(phi)).T # normalize the exponentials 
                assert np.all(np.round(np.sum(phi),5)) == 1, "emission probabilities don't sum to 1!"
        
        return phi