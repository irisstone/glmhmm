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
from warnings import simplefilter
import glmhmm.observations as obs

class GLM(object):
    
    """
    Base class for fitting generalized linear models. 
    Notation: 
        n: number of data/time points
        d: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        x: design matrix (nxm)
        y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)
    """
    def __init__(self,n,d,c,observations="bernoulli"):
        self.n, self.d, self.c = n, d, c
        
        # Master list of observation classes
        observation_classes = dict(
            bernoulli = obs.BernoulliObservations,
            multinomial = obs.MultinomialObservations
            )
        
        if observations not in observation_classes:
            raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))
 
        self.observations = observation_classes[observations](n=self.n,d=self.d,c=self.c)
        
    def compObs(self,x,w,normalize=True):
        """
        Computes the GLM observation probabilities for each data point.
        Parameters
        ----------
        x : nxd array of the data (design matrix)
        w : dxc array of weights
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
        
        w_init = np.zeros((self.d,self.c)) # initialize array
        w_init[:,1:] = np.random.uniform(low=wdist[0], high=wdist[1],size=(self.d,self.c-1)) # leave first column of weights zeros; randomly sample the rest 
        
        return w_init
    
    def generate_data(self,wdist=(-1,1),xdist=(-10,10)):
        
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
        x : nxd array of the data (design matrix)
        w : dxc array of weights
        y : nxc 1/0 array of observations
        """
        
        ## generate weights
        w = np.round(self.init_weights(wdist=wdist),2)

        ## generate data
        x = np.random.uniform(xdist[0], high=xdist[1],size=(self.n,self.d)) # choose length random inputs between -10 and 10
        
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

    def generate_data_from_fit(self, w, x, obs_ix, replace_obs=False, sessions=[None]):

        """
        Generate simulated data (design matrix amd observations) from fitted GLM weights                                                     
        Parameters
        ----------
        w : 1xd array
            fitted GLM weights
        x : nxd array of the true design matrix
            used for replicating the same environmental context (values other than past observations)
        obs_ix: list or tuple, optional
            includes indices of the first and last columns in the design matrix that are associated with observations
        replace_obs: boolean, optional
            determines whether or not to replace observation-related values in the design matrix with simulated ones (if there are no observation-related values
            in the design matrix, set to False)
        sessions: list, optional
            the indices of new sessions (if applicable) so that previous observations are coded appropriately at session boundaries
        Returns
        -------
        x : nxd array of the simulated design matrix (will only differ from true design matrix if observations are included as regressors)
        y : nxc 1/0 array of simulated observations
        """

        y = np.zeros(self.n)

        # zero out past observations in design matrix (we will simulate new ones)
        if replace_obs == True:
            num_past_obs = obs_ix[1] - obs_ix[0]
            x[:,obs_ix[0]:obs_ix[1]] = np.zeros((self.n,num_past_obs))

        for i in range(self.n):

            if replace_obs == True and (i in sessions):
                count = -1

            if replace_obs == True and count < num_past_obs and count != -1:
                x[i,obs_ix[0]:obs_ix[0]+count+1] = np.flip(y[i-count-1:i])
                count += 1
            elif replace_obs == True and count != -1: 
                x[i,obs_ix[0]:obs_ix[1]] = np.flip(y[i-num_past_obs:i])
            else: 
                x[i,obs_ix[0]:obs_ix[1]] = np.zeros(num_past_obs)
                count += 1

            ## generate observation probabilities for time point i
            phi = self.observations.compObs(x[i,:],w)
            assert np.round(np.sum(phi),3) == 1, "observation probabilities don't add up to 1"

            # generate observation for time point i
            y[i] = np.random.choice(np.arange(0,self.c,1), p = phi)

        return x,y
        
    def neglogli(self,x,w,y,reshape_weights=False,gammas=None,gaussianPrior=0):
        """
        Calculate the total loglikelihood p(y|x)
        Parameters
        ----------
        x : nxd array of the data (design matrix)
        w : dxc array of weights
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
            w = np.reshape(w,(self.d,self.c-1)) # unflatten weights

        phi = self.observations.compObs(x,w,normalize=False) # compute observation probabilities (phi)
        
        if reshape_weights:
            phi = np.hstack((np.ones((len(phi),1)),phi))
            
        norm = np.sum(phi,axis=1) # get normalization constant 
        weightedObs = np.sum(np.multiply(phi,y),axis=1)
        log_pyx = np.log(weightedObs) - np.log(norm) # compute loglikelihood
        
        assert np.round(np.sum(np.divide(phi.T,norm),axis=0),3).all()==1, 'Sum of normalized probabilities does not equal 1!'
        
        if gammas is not None:
            log_pyx = np.multiply(gammas,log_pyx) # apply weighting factor to loglikelihood
        
        self.ll = np.round(np.sum(log_pyx),30) # np.round ensures value is stored as float and not ArrayBox
        
        return -np.sum(log_pyx) + (((1*gaussianPrior)**2)/2 * np.sum(w ** 2))
    
    def fit(self,x,w,y,compHess=False,gammas=None,gaussianPrior=0):
        """
         Use gradient descent to optimize weights
        Parameters
        ----------
        x : nxd array of the data (design matrix)
        w : dxc array of weights
        y : nxc 1/0 array of observations
        compHess : boolean, optional
            sets whether or not to compute the Hessian of the weight matrix. The default is False.
        gammas : vector of floats, optional
            An array of values to include as weighting factors on the loglikelihood. If None, does not apply any weighting 
            (equivalent to incluing a gamma array of all 1s). Default is None.
        Returns
        -------
        w_new : dxc array of updated weights
        phi : nxc array of the updated observation probabilities
        """
        
        # reshape y from vector of indices to one-hot encoded array for matrix operations in neglogli
        if len(y.shape) == 1:
            yint = y.astype(int)
            y = np.zeros((yint.shape[0], yint.max()+1))
            y[np.arange(yint.shape[0]),yint] = 1

        # optimize loglikelihood given weights
        w_flat = np.ndarray.flatten(w[:,1:]) # flatten weights for optimization 
        opt_log = lambda w: self.neglogli(x,w,y,reshape_weights=True,gammas=gammas,gaussianPrior=gaussianPrior) # calculate log likelihood 
        simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning generated by scipy
        OptimizeResult = optimize.minimize(value_and_grad(opt_log),w_flat, jac = "True", method = "L-BFGS-B")
       
        self.w = np.hstack((np.zeros((self.d,1)),np.reshape(OptimizeResult.x,(self.d,self.c-1)))) # reshape and update weights
        # Get updated observation probabilities 
        self.phi = self.observations.compObs(x,w) 
        
        if compHess:
            ## compute Hessian
            hess = hessian(opt_log) # function that computes the hessian
            H = hess(self.w[:,1:]) # gets matrix for w_hats
            self.variance = np.sqrt(np.diag(np.linalg.inv(H.T.reshape((self.d * (self.c-1),self.d * (self.c-1)))))) # calculate variance of weights from Hessian
        
        return self.w,self.phi