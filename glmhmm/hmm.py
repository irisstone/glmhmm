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
        
    def _forwardPass(self,y,A,phi,pi0=None):
        
        '''
        Computes forward pass of Expectation Maximization (EM) algorithm; first half of .
        
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
        
        # if not fitting initial state probabilities, initialize to ones
        if not pi0:
            pi0 = np.ones(self.k)
        
        # first time bin
        pxz = np.multiply(phi[:,int(y[0])],pi0) # weight t=0 observation probabilities by initial state probabilities
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
        
    
    def _backwardPass(self,y,A,phi,alpha,cs):
        
        '''
        Computes backward pass of Expectation Maximization (EM) algorithm; second half of "E-step".
        
        Parameters
        ----------
        y : nx1 vector of observations
        A : kxk matrix of transition probabilities
        phi : kxc or nxkxc matrix of emission probabilities
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
        
        return pBack,beta,zhatBack
    
    def _updateTransitions():
        
                # CALCULATE XIS
        xis_sum = []
        xis = np.zeros((nT-1,nStates,nStates))
        
        for i in np.arange(0,nT-1):
            if method == "GLM":
                bm = np.multiply(bb[i+1,:],M[:,i,int(yy[i+1])])
            elif method == "standard":
                bm = np.multiply(bb[i+1,:],M[int(yy[i+1]),:])
            a = np.reshape(aa[i,:],(nStates,1))
            abm = np.multiply(bm,a)
            abmA = np.multiply(abm,A)/cs[i+1]
        
            xis[i,:,:] = abmA
            xis_sum.append(np.sum(abmA))
            

        assert np.round(sum(gammas[0]),5) == 1, "Sum of gammas does not equal 1" 
        
        
        # new transition matrix
        xis_n = np.reshape(np.sum(xis,axis=0),(nStates,nStates)) # sum_N xis
        xis_kn = np.reshape(np.sum(np.sum(xis,axis=0),axis=1),(nStates,1)) # sum_k sum_N xis
        A_jk = xis_n/xis_kn
        
        return A
        
    def _updateObservations():
        
        x_ni = np.zeros((nT,nObs))
        for i in np.arange(nT):
            x_ni[i,int(yy[i])] = 1 # assign 1 to index assoc. with observation

            
        if method == "standard":
        
            # new observation matrix
            gammaxni_n = np.dot(gammas.T,x_ni).T # sum_N gamma(z_nk)*x_ni
            gamma_n = np.sum(gammas,axis=0) # sum_N gamma(z_nk)
            mu_ik = gammaxni_n/gamma_n
            
            M = mu_ik # observation matrix
            
        if method == "GLM":
            
            ## run GLM to get new observation matrix, weights, covariance of weights
            mu_ik = np.zeros((nStates,nT,nObs))
            w_updated = np.zeros((nStates,nFeat,nObs))
            variances = np.zeros((nStates,nFeat*(nObs-1)))
            for zn in np.arange(nStates):
                mu_ik[zn,:,:], w_updated[zn,:,:], logl, _ = GLM.weighted_optimize_weights(w_init[zn,:,:],x_true,y_true,gammas[:,zn],"L-BFGS-B",bias=0, compHess = False, gaussPrior = gaussian_prior) 
                
            M =  mu_ik # update array of observation matrices
            w_init = w_updated # update weights
        
        return phi
        
    def _updateInitStates(gammas):
        '''
        Computes the updated initial state probabilities as part of the M-step of the EM algorithm.

        Parameters
        ----------
        gammas : nxk matrix of the posterior probabilities of the latent states

        Returns
        -------
        pi0 : kx1 vector of initial latent state probabilities (for t=1)

        '''
        
        pi0 = np.divide(gammas[0],sum(gammas[0])) # new initial latent state probabilities
        
        return pi0
    
    def _updateParams(gammas,beta,alpha,cs,A,fit_init_states = False):
        
        A = _updateTransitions():
            
        phi = _updateObservations():
        
        pi0 = _updateInitStates(gammas)
        
        # Store values in matrices
        As[n,:,:] = A
        if method == "standard":
            Ms[n,:,:] = M
        elif method == "GLM":
            M_rearranged = np.zeros((nT,nObs,nStates))
            for i in np.arange(0,nT):
                M_rearranged[i,:,:] = M[:,i,:].T
            
            Ms[n,:,:,:] = M_rearranged
            
        ws[n,:,:,:] = w_updated
            
        logLikelihoods[n] = ll
        
        return A,phi,pi0

    
    def EM(self):
        
        return