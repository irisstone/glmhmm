#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:12:26 2020

@author: istone

Class for fitting hidden Markov models (HMMs).

Updated: Sep 1, 2020

"""

class HMM(object):

    """
    Base class for fitting hidden Markov models. 
    Notation: 
        n: number of data/time points
        m: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        k: number of states (states)
        x: design matrix (nxm)
        y: observations (nxc)
        w: weights mapping x to y (mxc or mx1)

    """
    
        def __init__(self,n,m,c):
            self.n, self.m, self.c, self.k  = n, m, c, k
            
        def initialize_transitions(self,distribution='dirichlet',alpha_diag=5,alpha_full=1):
            
            """
            Initializes values for the transition probabilities.
    
            Parameters
            ----------
            distribution : string, optional
                Sets the distribution to use when initializing the transition probabilities. The default is dirichlet.
            alpha_diag : int, optional
                Sets the concentration parameter for the diagonal values when using a Dirichlet distribution. Default is 5. 
            alpha_full : int, optional
                Sets the concentration parameter for the off-diagonal values when using a Dirichlet distribution. Default is 1.
    
            Returns
            -------
            A : kxk matrix of the transition probabilities
    
            """
            
            if distribution == 'dirichlet':
            
                # Make transition matrix by sampling each row from Dirichlet distribution (achieved by normalizing gamma random variables)
                A = np.random.gamma(alpha_full*np.ones((self.k,self.k)) + alpha_diag*np.identity(self.k),1)
                A = A/np.repeat(np.reshape(np.sum(A,axis=1),(1,self.k)),self.k,0).T
                
            elif distribution == 'uniform':
                
                # Make transition matrix probabilities uniform 
                A = (1/self.k) * np.ones((self.k,self.k))
            
            return A
            
        def initialize_params(self,alpha_diag,alpha_full,alpha_emissions,nStates,nObs,nT,method):
            
            return
            
        def generate_data(self,self.n,A,M):

            return
        
        def neglogli(self):
            
            return
            
        def forwardPass(self):
            
            return
        
        def backwardPass(self):
            
            return
        
        def EM(self):
            
            return