#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:11:13 2020

@author: istone

Functions for initializing the parameters of a hidden Markov Model (\theta = {A,\phi,\pi}) using different distributions. 
Current supported distributions: Dirichlet, uniform. Also includes support for custom distributions. 
"""
import numpy as np

def init_transitions(self,distribution='dirichlet',alpha_diag=5,alpha_full=1):
    
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

def init_states(self,distribution='uniform'):
    
    """
    Initializes values for the state probabilities at t=1.

    Parameters
    ----------
    distribution : string, optional
        Sets the distribution to use when initializing the state probabilities for t=1. The default is uniform (same probability for each state).

    Returns
    -------
    A : kx1 vector of the state probabilities for t=1

    """
    
    if distribution == 'uniform':
        pi = (1/self.k) * np.ones((self.k,1))
    
    return pi

def init_emissions(self,distribution='dirichlet',alpha_diag=5,alpha_full=1):
    
    """
    Initializes values for the emission (observation) probabilities.

    Parameters
    ----------
    distribution : string, optional
        Sets the distribution to use when initializing the emission probabilities. The default is dirichlet.
    alpha_diag : int, optional
        Sets the concentration parameter for the diagonal values when using a Dirichlet distribution. Default is 5. 
    alpha_full : int, optional
        Sets the concentration parameter for the off-diagonal values when using a Dirichlet distribution. Default is 1.

    Returns
    -------
    phi : nxc matrix of the emission probabilities

    """
    
    if distribution == 'dirichlet':
        # Make emissions matrix by sampling each row from Dirichlet distribution (achieved by normalizing gamma random variables)
        phi = np.random.gamma(alpha_full,1,(self.k,self.c))
        if self.c > self.k: 
            phi = phi + np.append(alpha_diag*np.identity(self.k), np.zeros((self.k,self.c-self.k)),axis=1) # add to diagonal
        elif self.k > self.c: 
            phi = phi + np.append(alpha_diag*np.identity(self.c), np.zeros((self.k-self.c,self.c)),axis=0) # add to diagonal
        elif self.k == self.c:
            pass

        phi = phi/(np.repeat(np.reshape(np.sum(phi,axis=1),(1,self.k)),self.c,0).T) # normalize so columns sum to 1
    
    return phi