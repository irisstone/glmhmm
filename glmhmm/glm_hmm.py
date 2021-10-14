#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:50:03 2021

@author: irisstone
"""

import numpy as np
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
        
        # reshape y from vector of indices to one-hot encoded array for matrix operations in glm.fit
        yint = y.astype(int)
        yy = np.zeros((yint.shape[0], yint.max()+1))
        yy[np.arange(yint.shape[0]),yint] = 1
        
        self.phi = np.zeros((self.n,self.k,self.c))
        
        for zk in np.arange(self.k):
            self.w[zk,:,:], self.phi[:,zk,:] = self.glm.fit(x,w[zk,:,:],yy,compHess=self.hessian,gammas=gammas[:,zk],gaussianPrior=self.gaussianPrior)
            
            
        return self.w, self.phi
    
    def _updateParams(self,y,x,gammas,beta,alpha,cs,A,phi,w,fit_init_states = False):
        '''
        Computes the updated parameters as part of the M-step of the EM algorithm.

        Parameters
        ----------
        y : nx1 vector of observations
        gammas : nxk matrix of the posterior probabilities of the latent states
        beta : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities
        fit_init_states : boolean indicating whether initial state distribution is included as a learned parameter

        Returns
        -------
        kx1 vector of initial latent state probabilities (for t=1)

        '''
        
        A = self._updateTransitions(y,alpha,beta,cs,A,phi)
            
        w, phi = self._updateObservations(y,x,w,gammas)
        
        if fit_init_states: 
            self.pi0 = self._updateInitStates(gammas)
        
        return A, w, phi, self.pi0
    
    def fit(self,y,x,A,w,pi0=None,fit_init_states=False,maxiter=250,tol=1e-3,sess=None,B=1):
        '''

        Parameters
        ----------
        y : nx1 vector of observations 
        A : initial kxk matrix of transition probabilities
        phi : initial kxc or nxkxc matrix of emission probabilities
        pi0 : initial kx1 vector of state probabilities for t=1.
        fit_init_states : boolean, determines if EM will including fitting pi
        maxiter : int. The maximum number of iterations of EM to allow. The default is 250.
        tol : float. The tolerance value for the loglikelihood to allow early stopping of EM. The default is 1e-3.
        sessions : an optional vector of the first and last indices of different sessions in the data (for
        separate computations of the E step; first and last entries should be 0 and n, respectively)  
        B : an optional temperature parameter used when fitting via direct annealing EM (DAEM; see Ueda and Nakano 1998)                                                                                         
        Returns
        -------
        lls : vector of loglikelihoods for each step of EM, size maxiter 
        A : fitted kxk matrix of transition probabilities
        w : fitted kxdxc omatrix of weights
        pi0 : fitted kx1 vector of state probabilities for t= (only different from initial value of fit_init_states=True)

        '''
        
        lls = np.empty(maxiter)
        lls[:] = np.nan
            
        # store variables
        self.pi0 = pi0
        
        # # compute phi for each state from weights 
        phi = np.zeros((self.n,self.k,self.c))
        for i in range(self.n):
            for zi in range(self.k):
                phi[i,zi,:] = self.glm.observations.compObs(x[i,:],w[zi,:,:])
        
        # if emissions matrix is kxc (not input driven), add dimension along n for consistency in later code
        # phi = np.array([[0.19292068, 0.80707932],[0.95734311, 0.04265689]])
        # if len(phi.shape) == 2:
        #     phi = phi[np.newaxis,:,:] # add axis along n dim
        #     phi = np.tile(phi, (self.n,1,1)) # stack matrix n times
        
        if sess is None:
            sess = np.array([0,self.n]) # equivalent to saying the entire data set has one session
        
        for n in range(maxiter):
            
            # E STEP
            alpha = np.zeros((self.n,self.k))
            beta = np.zeros_like(alpha)
            cs = np.zeros((self.n))
            pBack = np.zeros_like(alpha)
            zhatBack = np.zeros_like(cs)
            ll = 0
            
            for s in range(len(sess)-1): # compute E step separately over each session or day of data 
                ll_s,alpha_s,cs_s = self.forwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:,:],pi0=pi0)
                pBack_s,beta_s,zhatBack_s = self.backwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:,:],alpha_s,cs_s)
                
                
                ll += ll_s
                alpha[sess[s]:sess[s+1]] = alpha_s
                cs[sess[s]:sess[s+1]] = cs_s
                pBack[sess[s]:sess[s+1]] = pBack_s ** B
                beta[sess[s]:sess[s+1]] = beta_s
                zhatBack[sess[s]:sess[s+1]] = zhatBack_s
                
            
            lls[n] = ll
            
            # M STEP
            A,w,phi,pi0 = self._updateParams(y,x,pBack,beta,alpha,cs,A,phi,w,fit_init_states = fit_init_states)
            
            
            # CHECK FOR CONVERGENCE    
            lls[n] = ll
            if  n > 0 and lls[n-1] + tol >= ll: # break early if tolerance is reached
                break
        
        return lls,A,w,pi0