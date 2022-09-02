#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:51:54 2022

@author: tanapanda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Det_NCG_Mdl:
    def __init__(self,
                 sigma = 2.000, # risk aversion
                 beta  = 0.960, # discount factor
                 theta = 0.330, # capital share in production function
                 delta = 0.081, # depreciation rate
                 A     = 0.592  # TFP
                 ):
        self.sigma = sigma
        self.beta  = beta
        self.theta = theta
        self.delta = delta
        self.A     = A
        
    def SteadyState(self, PrintResult = True):
        beta  = self.beta
        theta = self.theta
        delta = self.delta
        A     = self.A
       
        # Calculate the steady-state investment
        k_ss = A**(-theta) * theta**(-(1+theta)) * (1-theta) * \
            (1/beta + delta - 1)**(1+theta)
        k_ss = k_ss**(1/(theta**2 - theta - 1))
        
        
        # Calculate the steady-state labor
        l_ss = (A * (1-theta))**(1/(1+theta)) * k_ss**(theta/(1+theta)) 
        
        # Calculate the steady-state consumption
        c_ss = A * k_ss**theta * l_ss**(1-theta) - delta*k_ss
        
        # Store the steady-state values as attributes
        self.k_ss = k_ss
        self.l_ss = l_ss
        self.c_ss = c_ss
        
        if PrintResult:
            print(\
                  "\n",
                  "****** [Steady State Values] *****",\
                  "\n  k_ss = {:.4f}".format(k_ss),\
                  "\n  l_ss = {:.4f}".format(l_ss),\
                  "\n  c_ss = {:.4f}".format(c_ss),\
                  "\n **********************************"
                  )
        
        
