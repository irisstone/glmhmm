#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:50:03 2021

@author: irisstone
"""

from glmhmm.hmm import HMM
from glmhmm.init_params import init_transitions, init_states, init_weights
from glmhmm import glm

class GLMHMM(HMM):
    
    def __init__(self,n,d,c,k,observations="bernoulli",hessian=False,gaussianPrior=0):
        
        super().__init__(n,d,c,k)
        
        self.hessian, self.gaussianPrior = hessian, gaussianPrior
        
        self.glm = glm.GLM(self.n,self.d,self.c,observations=observations)
        
    def generate_params(self,weights=['uniform',-1,1,1],transitions=['dirichlet',5,1],state_priors='uniform'):
        
        '''
        Generates parameters A, w, and pi for a GLM-HMM. Can be used to generate true parameters for simulated data
        or to initialize parameters for fitting. 
        
        Parameters
        ----------
        weights : list, optional
            Contains the name of the desired distribution (string) and optionally the associated parameters 
            (see init_params.py script for details. The default is ['uniform',-1,1,1].
        transitions : list, optional
            Contains the name of the desired distribution (string). The default is ['dirichlet',5,1].
        state_priors : string, optional
            Containts the name of the desired distribution (string). The default is None, or 'uniform'.

        Returns
        -------
        A : kxk matrix of transition probabilities.
        w : mxc matrix of weights.
        pi : kx1 vector of state probabilities for t=1.

        '''
        
        A = init_transitions(self,distribution=transitions[0],alpha_diag=transitions[1],alpha_full=transitions[2])
        
        # initialize using different distributions or by fitting to a GLM and adding noise
        w = init_weights(self,distribution=weights[0],params=weights[1:-1],bias=weights[-1])
        
        pi = init_states(self,state_priors)
        
        return A, w, pi
        
    def generate_data(self,A,w):
        '''

        Parameters
        ----------
        A : kxk matrix of transition probabilities
        w : kxdxc matrix of eweights

        Returns
        -------
        y : nx1 vector of observations (the data)
        z : nx1 vector of latent states
        x : nxd matrix of inputs

        '''
        
        zi = np.random.choice(np.arange(0,len(A)))  # randomly select initial state
        y = np.zeros(self.n) 
        z = np.zeros(self.n)
        phi = np.zeros((self.n,self.k,self.c))
        
        # generate inputs
        x = np.random.randint(-10, high=10,size=(self.n,self.d)) # choose length random inputs between -10 and 10
        
        # generate observations and states using A and phi
        for i in range(self.n):
            z[i] = zi
            
            # compute phi for given state from weights 
            phi = self.glm.observations.compObs(x[i,:],w[zi,:,:])
            

            # select z_{i+1} using z_i and A
            zi = np.random.choice(A.shape[0], p = A[zi, :])
            
            # generate y's using probabilities from chosen latent state at each time point
            y[i] = np.random.choice(self.c, p = phi)
        
        return y, z, x
    
            
    def _updateObservations(self,y,x,w,gammas):
        '''
        Updates emissions probabilities as part of the M-step of the EM algorithm.
        For stationary observations, see the HMM class
        Uses gradient descent to find optimal update of weights
        
        Parameters
        ----------
        y : nx1 vector of observations
        gammas : nxk matrix of the posterior probabilities of the latent states
        
        Returns
        -------
        kxc matrix of updated emissions probabilities

        '''
        
        for zk in np.arange(HMM.k):
            self.w[zk,:,:], self.phi[:,zk,:] = self.glm.fit(x,w[zk,:,:],y,compHess=self.hessian,gammas=gammas[:,zk],gaussianPrior=self.gaussianPrior)
            
            
        