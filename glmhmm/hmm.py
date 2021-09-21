#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:12:26 2020

@author: istone

Class for fitting hidden Markov models (HMMs).

Updated: Sep 1, 2020

"""

import numpy as np
from glmhmm.init_params import init_transitions, init_emissions, init_states, init_weights
from glmhmm import glm

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
    
    
    def generate_params(self,emissions=['dirichlet',5,1],transitions=['dirichlet',5,1],state_priors='uniform'):
        '''
        Generates parameters A, phi, and pi for an HMM. Can be used to generate true parameters for simulated data
        or to initialize parameters for fitting. 
        
        Parameters
        ----------
        emissions : list, optional
            Contains the name of the desired distribution (string). The default is ['uniform',5,1].
        transitions : TYPE, optional
            DESCRIPTION. The default is ['dirichlet',5,1].
        state_priors : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        A : kxk matrix of transition probabilities.
        phi : mxc matrix of emission probabilities.
        pi : kx1 vector of state probabilities for t=1.

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
            p = phi[zi,:] # for static emission probabilities

            y[i] = np.random.choice(phi.shape[1], p = p)

            # select z_{i+1} using z_i and A
            zi = np.random.choice(A.shape[0], p = A[zi, :])
        
        return y, z

    
    def neglogli(self):
        
        return
        
    def forwardPass(self,y,A,phi,pi0=None):
        
        '''
        Computes forward pass of Expectation Maximization (EM) algorithm; first half of E-step.
        
        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : nxkxc matrix of emission probabilities

        Returns
        -------
        ll : float, marginal log-likelihood of the data p(y)
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods

        '''
        
        alpha = np.zeros((self.n,self.k)) # forward probabilities p(z_t | x_1:t)
        cs = np.zeros(self.n) # forward marginal likelihoods
        
        # if not fitting initial state probabilities, initialize to ones
        if not np.any(pi0):
            pi0 = np.ones(self.k)/self.k
            
        # if phi is 2d, add a time/trial axis (repeats matrix n times for stationary transitions)
        if len(phi.shape) == 2:
            phir = np.broadcast_to(phi, (self.n, self.k, self.c))
        elif len(phi.shape) == 3:
            phir = phi
        
        # first time bin
        pxz = np.multiply(phir[0,:,int(y[0])],np.squeeze(pi0)) # weight t=0 observation probabilities by initial state probabilities
        cs[0] = np.sum(pxz) # normalizer
        alpha[0] = pxz/cs[0] # conditional p(z_1 | x_1)
    
        # forward pass for remaining time bins
        for i in np.arange(1,self.n):
            alpha_prior = alpha[i-1]@A # propogate uncertainty forward
            pxz = np.multiply(phir[i,:,int(y[i])],alpha_prior) # joint P(y_1:t,z_t)
            cs[i] = np.sum(pxz) # conditional p(y_t | y_1:t-1)
            alpha[i] = pxz/cs[i] # conditional p(z_t | y_1:t)
        
        ll = np.sum(np.log(cs))
        
        return ll,alpha,cs
        
    
    def backwardPass(self,y,A,phi,alpha,cs):
        
        '''
        Computes backward pass of Expectation Maximization (EM) algorithm; second half of "E-step".
        
        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : nxkxc matrix of emission probabilities
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods

        Returns
        -------
        pBack : nxk matrix of the posterior probabilities of the latent states
        beta : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        zhatBack : nx1 vector of the most probable state at each time point

        '''
        
        beta = np.zeros((self.n,self.k))
        
        # last time bin
        beta[-1] = 1 # take beta(z_N) = 1
        
        # backward pass for remaining time bins
        for i in np.arange(self.n-2,-1,-1):
            beta_prior = np.multiply(beta[i+1],phi[:,int(y[i+1])]) # propogate uncertainty backward
            beta[i] = (A@beta_prior)/cs[i+1]
            
        pBack = np.multiply(alpha,beta) # posterior after backward pass -> alpha_hat(z_n)*beta_hat(z_n)
        zhatBack = np.argmax(pBack,axis=1) # decode from likelihoods only
        
        assert np.round(sum(pBack[0]),5) == 1, "Sum of posterior state probabilities does not equal 1"
        
        return pBack,beta,zhatBack
    
    def _updateTransitions(self,y,alpha,beta,cs,A,phi):
        
        '''
        Updates transition probabilities as part of the M-step of the EM algorithm.
        Currently only functional for stationary transitions (GLM on transitions not supported)
        Uses closed form updates as described in Bishop Ch. 13
        
        Parameters
        ----------
        y : nx1 vector of observations
        alpha : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        beta : nx1 vector of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        cs : nx1 vector of the forward marginal likelihoods
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities
        

        Returns
        -------
        A_new : kxk matrix of updated transition probabilities

        '''
        
        # compute xis, the joint posterior distribution of two successive latent variables p(z_{t-1},z_t |Y,theta_old)
        xis = np.zeros((self.n-1,self.k,self.k))
        for i in np.arange(0,self.n-1):
            beta_phi = beta[i+1,:] * phi[i,int(y[i+1]),:]
            alpha_reshaped = np.reshape(alpha[i,:],(self.k,1))
            xis[i,:,:] = ((beta_phi * alpha_reshaped) * A)/cs[i+1]
        
        # reshape and sum xis to obtain new transition matrix
        xis_n = np.reshape(np.sum(xis,axis=0),(self.k,self.k)) # sum_N xis
        xis_kn = np.reshape(np.sum(np.sum(xis,axis=0),axis=1),(self.k,1)) # sum_k sum_N xis
        A_new = xis_n/xis_kn
        
        return A_new
        
    def _updateObservations(self,y,gammas):
        '''
        Updates emissions probabilities as part of the M-step of the EM algorithm.
        For input-driven observations, see the GLMHMM class
        Uses closed form updates as described in Bishop Ch. 13 p. 618
        
        Parameters
        ----------
        y : nx1 vector of observations
        gammas : nxk matrix of the posterior probabilities of the latent states
        
        Returns
        -------
        kxc matrix of updated observation probabilities

        '''
            
        x_ni = np.zeros((self.n,self.c))
        for i in np.arange(self.n):
            x_ni[i,int(y[i])] = 1 # assign 1 to index assoc. with observation
        
        return (gammas.T @ x_ni).T / np.sum(gammas,axis=0)
        
    def _updateInitStates(self,gammas):
        '''
        Computes the updated initial state probabilities as part of the M-step of the EM algorithm.

        Parameters
        ----------
        gammas : nxk matrix of the posterior probabilities of the latent states

        Returns
        -------
        kx1 vector of initial latent state probabilities (for t=1)

        '''
        
        return np.divide(gammas[0],sum(gammas[0])) # new initial latent state probabilities
    
    def _updateParams(self,y,gammas,beta,alpha,cs,A,phi,fit_init_states = False):
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
            
        phi = self._updateObservations(y)
        
        if fit_init_states: 
            pi0 = self._updateInitStates(gammas)
        else:
            pi0 = self.pi0
        
        return A, phi, pi0

    
    def fit(self,y,A,phi,pi0=None,fit_init_states=False,maxiter=250,tol=1e-3):
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

        Returns
        -------
        lls : vector of loglikelihoods for each step of EM, size maxiter 
        A : fitted kxk matrix of transition probabilities
        phi : fitted kxc or nxkxc matrix of emission probabilities
        pi0 : fitted kx1 vector of state probabilities for t= (only different from initial value of fit_init_states=True)

        '''
        
        lls = np.nans(maxiter)
        
        for n in range(maxiter):
            # E STEP
            ll,alpha,cs = HMM.forwardPass(self,y,A,phi,pi0=pi0)
            pBack,beta,zhatBack = HMM.backwardPass(self,y,A,phi,alpha,cs)
            
            
            # M STEP
            A,phi,pi0 = HMM._updateParams(self,y,pBack,beta,alpha,cs,A,phi,fit_init_states = fit_init_states)
            
            # if emissions matrix is kxc (not input driven), add dimension along n for consistency in later code
            if len(phi.shape) == 2:
                phi = phi[np.newaxis,:,:] # add axis along n dim
                phi = np.tile(phi, (self.n,1,1)) # stack matrix n times
                
                
            lls[n] = lls
            if  n > 0 and ll[n-1] + tol >= ll: # break early if tolerance is reached
                break
        
        return lls,A,phi,pi0
    
class GLMHMM(HMM):
    
    def __init__(self,n,d,c,k,observations="bernoulli",hessian=False,gaussianPrior=0):
        
        super().__init__(n,d,c,k)
        
        self.hessian, self.gaussianPrior = hessian, gaussianPrior
        
        self.glm = glm.GLM(HMM.n,HMM.m,HMM.c,observations=observations)
        
    def generate_params(self,weights=['uniform',-1,1],transitions=['dirichlet',5,1],state_priors='uniform'):
        
        '''
        Generates parameters A, w, and pi for a GLM-HMM. Can be used to generate true parameters for simulated data
        or to initialize parameters for fitting. 
        
        Parameters
        ----------
        weights : list, optional
            Contains the name of the desired distribution (string) and optionally the associated parameters. The default is ['uniform',-1,1].
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
        w : kxmxc matrix of eweights

        Returns
        -------
        y : nx1 vector of observations (the data)
        z : nx1 vector of latent states
        x : nxm matrix of inputs

        '''
        
        zi = np.random.choice(np.arange(0,len(A)))  # randomly select initial state
        y = np.zeros(self.n) 
        z = np.zeros(self.n)
        phi = np.zeros(HMM.n,HMM.k,HMM.c)
        
        # generate inputs
        x = np.random.randint(-10, high=10,size=(self.n,self.m)) # choose length random inputs between -10 and 10
        
        # generate observations and states using A and phi
        for i in range(self.n):
            z[i] = zi
            
            # compute phi from weights 
            phi[:,zi,:] = self.glm.observations.compObs(x,w)

            # select z_{i+1} using z_i and A
            zi = np.random.choice(A.shape[0], p = A[zi, :])
            
            # generate y's using probabilities from chosen latent state at each time point
            y[i] = np.random.choice(HMM.c, p = phi[:,zi,:])
        
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
            
            
        