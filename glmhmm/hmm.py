#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:12:26 2020

@author: istone

Class for fitting hidden Markov models (HMMs).

Updated: Sep 1, 2020

"""

class HMM(object):
    
        def __init__(self,n,m,c):
            self.n, self.m, self.c = n, m, c