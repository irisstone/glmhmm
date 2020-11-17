#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:12:26 2020

@author: istone

Class for fitting hidden Markov models (HMMs).

Updated: Sep 1, 2020

"""

import numpy as np
from glmhmm.init_params import init_transitions, init_emissions, init_states

class HMM(object):

    """
    Base class for fitting hidden Markov models. 
    Notation: 
        n: number of data points
        d: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        k: number of states (states)
        X: design matrix (nxm)
        Y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)

    """
    
    def __init__(self,n,d,c,k):
            self.n, self.d, self.c, self.k  = n, d, c, k
    
    def initialize_parameters(self,emissions=['dirichlet',5,1],transitions=['dirichlet',5,1],state_priors='uniform'):
        '''

        Parameters
        ----------
        emissions : list, optional
            Contains the name of the desired distribution for initialization (string). The default is ['uniform',5,1].
        transitions : TYPE, optional
            DESCRIPTION. The default is ['dirichlet',5,1].
        state_priors : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        A : kxk matrix of initial transition probabilities.
        phi : mxc matrix of initial emission probabilities.
        pi : kx1 vector of initial state probabilities for t=1.

        '''
        
        A = init_transitions(self,distribution=transitions[0],alpha_diag=transitions[1],alpha_full=transitions[2])
        
        phi = init_emissions(self,distribution=emissions[0],alpha_diag=emissions[1],alpha_full=emissions[2])
        
        pi = init_states(self,state_priors)
        
        return A,phi,pi
    

        
    def generate_data(self,A,phi):
        '''

        Parameters
        ----------
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities

        Returns
        -------
        y : nx1 vector of observations (the data)
        z : nx1 vector of latent states

        '''
        
        zi = np.random.choice(np.arange(0,len(A)))  # randomly select initial state
        y = np.zeros(self.n) 
        z = np.zeros(self.n)
        
        # generate observations and states using A and phi
        for i in range(self.n):
            z[i] = zi
            #if len(M.shape) == 2:
            p = phi[zi,:] # for static emission probabilities
            #elif len(M.shape) == 3:
                #p = M[i,zi,:] # for emission probabilities generated using a GLM

            y[i] = np.random.choice(phi.shape[1], p = p)

            # select z_{i+1} using z_i and A
            zi = np.random.choice(A.shape[0], p = A[zi, :])
        
        return y, z

    
    def neglogli(self):
        
        return
        
    def _forwardPass(self,y,A,phi):
        
        '''
        Computes forward pass of Expectation Maximization (EM) algorithm.
        
        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities

        Returns
        -------
        ll : float, marginal log-likelihood of the data p(y)
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods

        '''
        
        alpha = np.zeros((self.n,self.k)) # forward probabilities p(z_t | x_1:t)
        cs = np.zeros(self.n) # forward marginal likelihoods
        
        # first time bin
        pxz = phi[:,int(y[0])]
        cs[0] = np.sum(pxz) # normalizer
        alpha[0] = pxz/cs[0] # conditional p(z_1 | x_1)
    
        # forward pass for remaining time bins
        for i in np.arange(1,self.n):
            alpha_prior = alpha[i-1]@A # propogate uncertainty forward
            pxz = np.multiply(phi[:,int(y[i])],alpha_prior) # joint P(x_1:t,z_t)
            cs[i] = np.sum(pxz) # conditional p(x_t | x_1:t-1)
            alpha[i] = pxz/cs[i] # conditional p(z_t | x_1:t)
        
        ll = np.sum(np.log(cs))
        
        return ll,alpha,cs
        
    
    def _backwardPass(self):
        
        return
    
    def EM(self):
        
        return