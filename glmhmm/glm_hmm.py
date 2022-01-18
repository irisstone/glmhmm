#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:50:03 2021

@author: irisstone
"""

import numpy as np
import autograd.numpy as npa
from autograd import hessian
from glmhmm.hmm import HMM
from glmhmm.init_params import init_transitions, init_states, init_weights
from glmhmm import glm
from glmhmm.utils import replace_inputs

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
        w : kxdxc matrix of weights

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

    def generate_data_from_fit(self, w, A, x, obs_ix=[0,0], replace=False, sessions=[0],outcomes=[None]):

        """
        Generate simulated data (design matrix amd observations) from fitted GLM weights                                                     
        Parameters
        ----------
        w : 1xd array
            fitted GLM-HMM weights
        A : kxk array
            fitted transition probabilities
        x : nxd array of the true design matrix
            used for replicating the same environmental context (values other than past observations)
        obs_ix : list or tuple, optional
            includes indices of the first and last columns in the design matrix that are associated with observations,
            followed by the index associated with any outcome-based metric (e.g. previous rewarded choice).
            Should be set if replace=True. If replacing an outcome-based metric but not observations, set the first
            two indices to zero. 
        replace : boolean, optional
            determines whether or not to replace values in the design matrix, e.g. with simulated observations (if there are no observation-related values
            in the design matrix, set to False)
        sessions : list, optional
            the indices of new sessions (if applicable) so that previous observations are coded appropriately at session boundaries
        outcomes : list or vector, optional
            the correct outcome at each trial/time point, useful if outcome cannot be entirely decoded from other inputs
                (e.g. if stimuli is sometimes neutral or zero and correct outcomes are decided randomly in those instances)
        Returns
        -------
        x : nxd array of the simulated design matrix (will only differ from true design matrix if observations are included as regressors)
        y : nx1 1/0 array of simulated observations
        z : nx1 array of simulated state assignments
        """

        # initialize empty vectors
        y = np.zeros(self.n)
        z = np.zeros(self.n)
        num_past_obs = obs_ix[1] - obs_ix[0]

        z[0] = np.random.choice(np.arange(0,self.k))  # define initial state
        
        for i in range(self.n):
            if replace: 
                x = replace_inputs(x,y,outcomes,i,obs_ix)
            if replace and (i in sessions): # insert zeros at session boundaries
                x[i,obs_ix[0]:obs_ix[1]] = np.zeros((num_past_obs))
                if len(obs_ix) == 3 and x.shape[1] != obs_ix[2]:
                    x[:,obs_ix[2]] = 0
                elif len(obs_ix) == 3 and x.shape[1] != obs_ix[2]:
                    x[:,-1] = 0

            ## generate observation probabilities for time point i using state at time point i
            phi = self.glm.observations.compObs(x[i,:],w[int(z[i]),:,:])
            assert np.round(np.sum(phi),3) == 1, "observation probabilities don't add up to 1"

            # generate observation for time point i
            y[i] = np.random.choice(np.arange(0,self.c,1), p = phi)

            # select z for next time point according to z and A
            if i<self.n-1:
                z[i+1] = np.random.choice(np.arange(0, self.k), p = A[int(z[i]), :])

        return x,y,z
    
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
        
        self.A = self._updateTransitions(y,alpha,beta,cs,A,phi)
            
        self.w, self.phi = self._updateObservations(y,x,w,gammas)
        
        if fit_init_states: 
            self.pi0 = self._updateInitStates(gammas)
        
        return self.A, self.w, self.phi, self.pi0
    
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
        
        self.lls = np.empty(maxiter)
        self.lls[:] = np.nan
            
        # store variables
        self.pi0 = pi0
        
        # # compute phi for each state from weights 
        phi = np.zeros((self.n,self.k,self.c))
        for i in range(self.n):
            for zi in range(self.k):
                phi[i,zi,:] = self.glm.observations.compObs(x[i,:],w[zi,:,:])
        
        if sess is None:
            sess = np.array([0,self.n]) # equivalent to saying the entire data set has one session
        
        for n in range(maxiter):
            
            # E STEP
            alpha = np.zeros((self.n,self.k))
            beta = np.zeros_like(alpha)
            cs = np.zeros((self.n))
            self.pStates = np.zeros_like(alpha)
            self.states = np.zeros_like(cs)
            ll = 0
            
            for s in range(len(sess)-1): # compute E step separately over each session or day of data 
                ll_s,alpha_s,_,cs_s = self.forwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:,:],pi0=pi0)
                pBack_s,beta_s,zhatBack_s = self.backwardPass(y[sess[s]:sess[s+1]],A,phi[sess[s]:sess[s+1],:,:],alpha_s,cs_s)
                
                
                ll += ll_s
                alpha[sess[s]:sess[s+1]] = alpha_s
                cs[sess[s]:sess[s+1]] = cs_s
                self.pStates[sess[s]:sess[s+1]] = pBack_s ** B
                beta[sess[s]:sess[s+1]] = beta_s
                self.states[sess[s]:sess[s+1]] = zhatBack_s
                
            
            self.lls[n] = ll
            
            # M STEP
            A,w,phi,pi0 = self._updateParams(y,x,self.pStates,beta,alpha,cs,A,phi,w,fit_init_states = fit_init_states)
            
            
            # CHECK FOR CONVERGENCE    
            self.lls[n] = ll
            if  n > 5 and self.lls[n-5] + tol >= ll: # break early if tolerance is reached
                break
        
        self.A,self.w,self.phi,self.pi0 = A,w,phi,pi0

        return self.lls,self.A,self.w,self.pi0
    
    def computeVariance(self,x,y,A,w,gaussPrior=0):
        
        '''
        Compute the variance for the fitted parameters A and w of a GLM-HMM
        
        Parameters
        ----------
        x : nxd matrix of inputs
        y : nx1 vector of observations
        A : fitted kxk matrix of transition probabilities
        w : fitted kxdxc omatrix of weights
        gaussPrior : integer specifying the sigma of a desired Gaussian prior on the loss to penalize large
        weight values. The default is 0 which corresponds to no prior.

        Returns
        -------
        variance : a vector containing the variances for the fitted parameters A and w

        '''

        def logLikelihood(params_flat,y,x):
    
           # unflatten A
           A = params_flat[0:self.k*(self.k-1)]
           A = npa.reshape(A,(self.k,self.k-1)) # reshape A vector into s x s-1 matrix
           A_last = 1 - npa.sum(A,axis=1).reshape(self.k,1) # calculate values of last column of A (1 - sum of other row values)
           A = npa.hstack((A,A_last)) # append new column to transition matrix
    
           # unflatten W
           w = params_flat[int(self.k * (self.k-1)):]
           w = npa.reshape(w,(self.k,self.d,self.c-1))
    
           phi = [] # must use list appending because autograd numpy doesn't support indexing
    
           for zi in range(self.k):
               w_new = npa.reshape(w[zi,:,:],(self.d,self.c-1))
               p = npa.exp(x@w_new) # get exponentials e^wTx
               p = npa.hstack((p,npa.ones((len(p),1))))
               norm = npa.sum(p,axis=1) # get normalization constant (sum of exponentials)
               phi.append(npa.divide(p.T,norm).T) # normalize the exponentials
    
           phi = npa.array(phi)
              
           aa = [] # forward probabilities p(z_t | x_1:t)
           cs = [] # forward marginal likelihoods
    
           # first time bin
           pxz = phi[:,0,int(y[0])]
           cs.append(npa.sum(pxz)) # normalizer
           aa.append(pxz/cs[0]) # conditional p(z_1 | x_1)
    
           # forward pass for remaining time bins
           for i in npa.arange(1,self.n):
               aaprior = npa.dot(aa[i-1],A) # propogate uncertainty forward
               pxz = npa.multiply(phi[:,i,int(y[i])],aaprior) # joint P(x_1:t,z_t)
               cs.append(npa.sum(pxz)) # conditional p(x_t | x_1:t-1)
               aa.append(pxz/cs[i]) # conditional p(z_t | x_1:t)
    
           cs = npa.array(cs)
    
           return -npa.sum(npa.log(cs)) + ((1/(2*(gaussPrior**2)) * npa.sum(w ** 2)))

        # vectorize parameters
        A_flat = A[:,0:self.k-1].flatten(order='C') # flattens column-wise ([first row, second row, third row,...])
        w_flat = w.flatten(order='C')
        params_flat = np.hstack((A_flat,w_flat))
    
        # create lambda function specializing parameter(s) to take derivative with respect to
        opt_log = lambda params_flat: logLikelihood(params_flat,y,x)
    
        hess = hessian(opt_log) # function that computes the hessian
        H = hess(params_flat) # get hessian matrix
    
        ## calculate variance of parameters from Hessian
        variance = np.sqrt(np.diag(np.linalg.inv(H)))
       
        return variance