#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW2.py

is the python class for the assignment #2 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 13, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn
from tabulate import tabulate
from copy import deepcopy
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

class GHH_Model:
    def __init__(self,
                 alpha  , # capital share in production function
                 beta   , # discount factor
                 theta  , # inverse Frisch elasticity
                 gamma  , # risk avrsion
                 omega  , # convexity of the depreciation func.
                 B      , # coefficient in the depreciation func.
                 A      , # TFP level or list for TFP development
                 sigma  , # s.d. of the stochastic process
                 lmbd   , # Autocorrelation of the stochastic process
                 nGrids = 150,   # # of grid points for k
                 k_min  = 0.000, # the lower bound of k
                 k_max  = 5.000  # the upper bound of k
                 ):
        k_grid = np.linspace(k_min, k_max, nGrids)
        if k_min == 0:
            # to avoid the degenerated steady state, do not use k = 0.
            k_grid[0] = 0.001 * k_grid[1]
        elif k_min < 0:
            raise Exception('The lower bound of k is expected greater than zero.')
            
        pi11   = 0.5 * (lmbd + 1)
        pi22   = 0.5 * (lmbd + 1)
        Theta = sigma  
        
        # Store the parameter values as instance attributes
        self.alpha  = alpha
        self.beta   = beta
        self.theta  = theta
        self.gamma  = gamma
        self.omega  = omega
        self.B      = B
        self.A      = A
        self.sigma  = sigma
        self.lmbd   = lmbd
        self.nGrids = nGrids
        self.k_grid = k_grid
        self.pi11   = pi11
        self.pi22   = pi22
        self.Theta  = Theta
        
        
    def find_argmax(self, 
                   k, # capital stock today 
                   eps, # shock realized today
                   h_min =  0.0, # lower bound of utilixation rate
                   h_max =  1.0, # upper bound of utilixation rate (Note h is in [0,1])
                   l_min =  0.0, # lower bound of labor input
                   l_max = 50.0  # upper bound of labor input
                   ):
        """
        find_argmax returns the optimal utilization rate (h_hat) and the labor
        input (l_hat) given the capital stock and investment-specific 
        technology shock today
        """
        # load necessary parameter values from instance attributes
        alpha  = self.alpha
        theta  = self.theta
        omega  = self.omega
        A      = self.A
        B      = self.B
        nGrids = self.nGrids
                
        # Prepare grid points for utilization rate
        h_grid = np.linspace(h_min, h_max, nGrids)
        if h_min == 0:
            h_grid[0] = 0.001 * h_grid[1]
        elif h_min < 0:
            raise Exception('The lower bound of h is expected greater than zero.')
        
        # Prepare grid points for labor input
        l_grid = np.linspace(l_min, l_max, nGrids)
        if l_min == 0:
            l_grid[0] = 0.001 * l_grid[1]
        elif l_min < 0:
            raise Exception('The lower bound of l is expected should be greater than zero.')
        
        # Convert the grids from 1-D form to 2-D form.
        # This enables matrix calculation, which prevents excess usage of for-loops
        h_grid = np.reshape(h_grid, (nGrids, 1))
        l_grid = np.reshape(l_grid, (1, nGrids))
        
        # Calculate the possible values of the objective function at once
        obj_func_val = (
                         A * (k*h_grid)**alpha * l_grid**(1-alpha)
                       + k * (1 - B * h_grid**omega / omega) * np.exp(-eps)
                       - l_grid**(1+theta) / (1+theta)
                       )

        # Find the maximum enty of obj_func_val
        # Convert the return of argmax method into the matrix coordinate
        idx0, idx1 = np.unravel_index(np.argmax(obj_func_val, axis = None), 
                                      obj_func_val.shape)
        
        h_hat = h_grid[idx0] # Pick up the optimal utilization rate
        l_hat = l_grid[idx1] # Pick up the optimal labor input
        
        return h_hat, l_hat
    

    def eval_value_func(self, 
                        k, # capital stock today 
                        eps, # shock realized today
                        V_tmrw, # tomorrow's values (depending on k')
                        penalty = -999.0, # punitive value if any violation happens
                        isPolicyIteration = False, # if true, conduct Howard's policy itereation
                        isMonotone = False, # if true, exploit monotonicity
                        isConcave = False, # if true, exploit concavity
                        ):
        """
        feval_value_func evaluates today's value based on the given
        capital stock today (k), the given investment-specific shock today
        (epsilon), and tomorrow's value (V'). Note that V' depends on the
        capital stock tomorrow (k'), which is optimally chosen today.
        """
        # load necessary parameter values from instance attributes
        alpha   = self.alpha
        beta    = self.beta
        theta   = self.theta
        omega   = self.omega
        gamma   = self.gamma
        A       = self.A
        B       = self.B
        pi11    = self.pi11
        pi22    = self.pi22
        nGrids  = self.nGrids
        k_grids = self.k_grids

        # Allocate memory for the vector of today's value
        V_td = np.ones(nGrids) * penalty
        
        # Find the optimal utilization rate and the optimal
        # labor input under the given k and epsilon
        h_hat, l_hat = self.FindArgmax(k, eps)
        
        # Calculate the optimal consumption depending on k',
        # under the given k and epsilon and the computed 
        # h_hat and l_hat
        c_list = (
                 A * (k*h_hat)**alpha * l_hat**(1-alpha)
                 - k_grids * np.exp(-eps)
                 + k * (1 - B * h_hat**omega / omega) * np.exp(-eps)
                )
        
        # Calculate the inside of brackets in the utility function
        u_list = c_list - (l_hat**(1+theta)) / (1+theta)
        
        # Check if the inside is positive
        # (Otherwise)     
        isComputable =  (u_list >= 0)

        u_list[isComputable] = 1/(1-gamma) * (u_list[isComputable])**(1-gamma)
        u_list[isComputable==False] = penalty
        
        
                # TODO!: COMPLETE THE EXPRESSION
        return h_hat, l_hat