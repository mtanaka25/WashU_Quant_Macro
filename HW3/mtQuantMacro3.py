#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW2.py

is the python class for the assignment #3 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep XX, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from copy import deepcopy
from tabulate import tabulate
from random import randrange, seed


class SIMModel:
    def __init__(self,
                 rho       = 0.9900,
                 var_eps   = 0.0425,
                 var_y20   = 0.2900,
                 n_grids   = 10,
                 Omega     = 3.0000,
                 age_range = (20, 65),
                 n_draws   = 1000,
                 ) -> None:
        # Prepare list of ages
        age_list = [i for i in range(age_range[0], age_range[1]+1, 1)]  
        
        self.rho       = rho
        self.var_eps   = var_eps
        self.sig_eps   = np.sqrt(var_eps)
        self.var_y20   = var_y20
        self.sig_y20   = np.sqrt(var_y20)
        self.n_grids   = n_grids
        self.Omega     = Omega
        self.age_list  = age_list
        self.n_samples = n_draws

    def tauchen_discretize(self,
                is_save_y_grid = True,
                is_save_trans_mat = True):
        # Prepare y gird points
        sig_y21 = np.sqrt(self.var_eps / ((1-self.rho)**2))
        y_N     = self.Omega * sig_y21
        y_grid  = np.linspace(-y_N, y_N, self.n_grids)        
        
        # Calculate the step size
        h = (1 * y_N)/(self.n_grids-1)
        
        # 
        trans_mat = [ 
            [self.tauchen_trans_mat_ij(i, j, y_grid, h) for j in range(self.n_grids)]
            for i in range(self.n_grids)]
        
        return y_grid, trans_mat
        
    
    def tauchen_trans_mat_ij(self, i, j, y_grid, h):
        if j == 0:
            trans_mat_ij = norm((y_grid[0] - self.rho*y_grid[i] + h/2)/self.sig_eps).cdf
        elif j == self.n_grids:
            trans_mat_ij = 1 - norm((y_grid[0] - self.rho*y_grid[i] - h/2)/self.sig_eps).cdf
        else:
            trans_mat_ij = ( norm((y_grid[0] - self.rho*y_grid[i] + h/2)/self.sig_eps).cdf
                           - norm((y_grid[0] - self.rho*y_grid[i] - h/2)/self.sig_eps).cdf)
        return trans_mat_ij
    
    def run_simulation(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        
        y20_samples = lognorm.rvs(self.sig_y20, size=self.n_samples)
        y20_samples = np.sort(y20_samples)
        
        Lorenz = np.cumsum(y20_samples)/np.sum(y20_samples)
        
        plt.plot(Lorenz, label='Lorenz curve for y_20')
         