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
import time
from warnings import simplefilter
import glmhmm.observations as obs

class GLM(object):
    
    """
    Base class for fitting generalized linear models. 
    Notation: 
        n: number of data/time points
        m: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        x: design matrix (nxm)
        y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)

    """
    def __init__(self,n,m,c,observations="bernoulli"):
        self.n, self.m, self.c = n, m, c
        
        # Master list of observation classes
        observation_classes = dict(
            bernoulli = obs.BernoulliObservations,
            multinomial = obs.MultinomialObservations
            )
        
        if observations not in observation_classes:
            raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))
 
        self.observations = observation_classes[observations](n=self.n,m=self.m,c=self.c)
        
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
        
        phi = np.exp(x@w) # get exponentials e^(wTx)
        if normalize:
            phi = np.divide(phi.T,np.sum(phi,axis=1)).T # normalize the exponentials 
        
        return phi
    
    def init_weights(self,wdist=(-0.2,1.2)):
        """
        Initialize weights from uniform distribution.
        
        Parameters
        ----------
        wdist : tuple, optional
                sets high and low uniform distribution limits for randomly sampling weight values. The default is (-0.2,1.2).

        Returns
        -------
        w_init: mxc array of weights with first column set to zeros
        """
        
        w_init = np.zeros((self.m,self.c)) # initialize array
        w_init[:,1:] = np.random.uniform(low=wdist[0], high=wdist[1],size=(self.m,self.c-1)) # leave first column of weights zeros; randomly sample the rest 
        
        return w_init
    
    def generate_data(self,wdist=(-0.2,1.2),xdist=(-10,10)):
        
        """
        Generate simulated data (design matrix, weights, and observations) for fitting a GLM                                                      

        Parameters
        ----------
        wdist : tuple, optional
                sets high and low uniform distribution limits for randomly sampling weight values. The default is (-0.2,1.2).
        xdist : tuple, optional
                sets high and low limits for randomly sampling integer data values. The default is (-10,10).
        bias : boolean, optional
               determines whether or not to add a bias to the data. The default is True.

        Returns
        -------
        x : nxm array of the data (design matrix)
        w : mxc array of weights
        y : nxc 1/0 array of observations

        """
        
        ## generate weights
        w = self.init_weights(wdist=wdist)
        
        ## generate data
        x = np.random.randint(xdist[0], high=xdist[1],size=(self.n,self.m)) # choose length random inputs between -10 and 10
        
        ## generate observation probabilities
        phi = self.observations.compObs(x,w) 
        
        # generate 1-D vector of observations for each n
        cumdist = phi.cumsum(axis=1) # calculate the cumulative distributions
        undist = np.random.rand(len(cumdist), 1) # generate set of uniformly distributed samples
        obs = (undist < cumdist).argmax(axis=1) # see where they "fit" in cumdist
        
        # convert to nxc matrix of binary values
        y = np.zeros((self.n,self.c))
        y[np.arange(self.n),obs] = 1
            
        return x,w,y
        
    def neglogli(self,x,w,y,reshape_weights=False,gammas=None,gaussianPrior=0):
        """
        Calculate the total loglikelihood p(y|x)

        Parameters
        ----------
        x : nxm array of the data (design matrix)
        w : mxc array of weights
        y : nxc 1/0 array of observations
        reshape_weights : boolean, optional. Sets whether or not to reshape weights and add column of ones to phi. 
            The default is False. Typically only True if weights have been flattened prior to calling function, e.g.
            in advance of performing gradient descent. 
        gammas : vector of floats, optional
            An array of values to include as weighting factors on the loglikelihood. If None, does not apply any weighting 
            (equivalent to incluing a gamma array of all 1s). Default is None.
        gaussianPrior : float, optional. 
            Sets the inverse variance of the Gaussian prior on the loglikelihood function. Default is 0, equivalent to no prior. 

        Returns
        -------
        negative sum of the loglikelihood of the observations (y) given the data (x)

        """
        
        if reshape_weights:
            w = np.reshape(w,(self.m,self.c-1)) # unflatten weights

        phi = self.observations.compObs(x,w,normalize=False) # compute observation probabilities (phi)
        
        if reshape_weights:
            phi = np.hstack((np.ones((len(phi),1)),phi))
            
        norm = np.sum(phi,axis=1) # get normalization constant 
        weightedObs = np.sum(np.multiply(phi,y),axis=1)
        log_pyx = np.log(weightedObs) - np.log(norm) # compute loglikelihood
        
        assert np.round(np.sum(np.divide(phi.T,norm),axis=0),3).all()==1, 'Sum of normalized probabilities does not equal 1!'
        
        if gammas is not None:
            log_pyx = np.multiply(gammas,log_pyx) # apply weighting factor to loglikelihood
        
        self.ll = np.round(np.sum(log_pyx),16) # np.round ensures value is stored as float and not ArrayBox
        
        return -np.sum(log_pyx) + (((1*gaussianPrior)**2)/2 * np.sum(w ** 2))
    
    def fit(self,x,w,y,compHess=False,gammas=None,gaussianPrior=0):
        """
         Use gradient descent to optimize weights

        Parameters
        ----------
        x : nxm array of the data (design matrix)
        w : mxc array of weights
        y : nxc 1/0 array of observations
        compHess : boolean, optional
            sets whether or not to compute the Hessian of the weight matrix. The default is False.
        gammas : vector of floats, optional
            An array of values to include as weighting factors on the loglikelihood. If None, does not apply any weighting 
            (equivalent to incluing a gamma array of all 1s). Default is None.

        Returns
        -------
        w_new : mxc array of updated weights
        phi : nxc array of the updated observation probabilities

        """
        
        # optimize loglikelihood given weights
        w_flat = np.ndarray.flatten(w[:,1:]) # flatten weights for optimization 
        opt_log = lambda w: self.neglogli(x,w,y,reshape_weights=True,gammas=gammas,gaussianPrior=gaussianPrior) # calculate log likelihood 
        simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning generated by scipy
        OptimizeResult = optimize.minimize(value_and_grad(opt_log),w_flat, jac = "True", method = "L-BFGS-B")
       
        self.w = np.hstack((np.zeros((self.m,1)),np.reshape(OptimizeResult.x,(self.m,self.c-1)))) # reshape and update weights
        # Get updated observation probabilities 
        self.phi = self.observations.compObs(x,w) 
        
        if compHess:
            ## compute Hessian
            hess = hessian(opt_log) # function that computes the hessian
            H = hess(self.w_new[:,1:]) # gets matrix for w_hats
            self.variance = np.sqrt(np.diag(np.linalg.inv(H.T.reshape((self.m * (self.c-1),self.m * (self.c-1)))))) # calculate variance of weights from Hessian
        
        return self.w,self.phi