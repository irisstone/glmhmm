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
        m: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        k: number of states (states)
        X: design matrix (nxm)
        Y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)

    """
    
    def __init__(self,n,m,c,k):
            self.n, self.m, self.c, self.k  = n, m, c, k
    
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
        y : nxc matrix of observations (the data)
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
        
    def forwardPass(self):
        
        return
    
    def backwardPass(self):
        
        return
    
    def EM(self):
        
        return