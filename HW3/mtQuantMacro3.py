#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW3.py

is the python class for the assignment #3 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 20, 2022 (Masaki Tanaka, Washington University in St. Louis)

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
                 n_samples = 1000,
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
        self.n_samples = n_samples

    def tauchen_discretize(self,
                is_save_y_grid = True,
                is_save_trans_mat = True):
        # Prepare y gird points
        sig_y21 = np.sqrt(self.var_eps / (1-self.rho**2))
        y_N     = self.Omega * sig_y21
        y_grid  = np.linspace(-y_N, y_N, self.n_grids)
        Y_grid  = np.exp(y_grid)

        # Calculate the step size
        h = (2 * y_N)/(self.n_grids-1)
        
        # 
        trans_mat = [ 
            [self.tauchen_trans_mat_ij(i, j, y_grid, h) for j in range(self.n_grids)]
            for i in range(self.n_grids)]

        self.Y_grid = Y_grid
        self.y_grid = y_grid
        self.trans_mat = trans_mat
        self.step_size = h

        return y_grid, trans_mat
        
    
    def tauchen_trans_mat_ij(self, i, j, y_grid, h):
        if j == 0:
            trans_mat_ij = norm.cdf((y_grid[j] - self.rho*y_grid[i] + h/2)/self.sig_eps)
        elif j == (self.n_grids-1):
            trans_mat_ij = 1 - norm.cdf((y_grid[j] - self.rho*y_grid[i] - h/2)/self.sig_eps)
        else:
            trans_mat_ij = ( norm.cdf((y_grid[j] - self.rho*y_grid[i] + h/2)/self.sig_eps)
                           - norm.cdf((y_grid[j] - self.rho*y_grid[i] - h/2)/self.sig_eps))
        return trans_mat_ij
    
    
    def run_simulation(self, fixed_seed=None):
        if fixed_seed != None:
            np.random.seed(fixed_seed)
        
        Y20_samples = lognorm.rvs(self.sig_y20, size=self.n_samples)
        Y20_samples = np.sort(Y20_samples)
        
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Derive and plot Lorenz curve for earnings itself (not log earning)
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        fig0 = plt.figure(figsize=(8, 6))
        plt.hist(Y20_samples, bins=25, density=True)
        fig0.savefig('Sample_histgram.png', dpi=300)

        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Derive and plot Lorenz curve for earnings itself (not log earning)
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Calculate the Lorenz curve
        Lorenz_Y20_original = np.cumsum(Y20_samples)/np.sum(Y20_samples)
        self.Lorenz_Y20_original = Lorenz_Y20_original 
                
        # Plot the Lorenz curve
        x = np.linspace(0, 1, self.n_samples)
                
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(x, x, '-', linewidth = 0.5, color = 'green', label='perfect equality')
        plt.plot(x, Lorenz_Y20_original, '-r',lw = 3, label='Lorenz curve for Y_20')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig1.savefig('Lorenz_Y20_original.png', dpi=300)

        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Calculate the probability of each grid point
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        h = self.step_size
        y20_samples = np.log(Y20_samples)
        
        m20 = [np.sum((y20_samples <= thr+h/2) & (y20_samples> thr-h/2))
                      /self.n_samples
                      for thr in self.y_grid[1:-1]]
        m20.insert(0, 
            sum((y20_samples <= self.y_grid[0]+h/2))/self.n_samples)
        m20.append(1 - sum(m20))
        
        fig2 = plt.figure(figsize=(8, 6))
        plt.bar(self.Y_grid, m20)

        fig2.savefig('PDF_Y20.png', dpi=300)

        self.m20 = m20
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Lorenz curve based on grid points 
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        m20_filtered = np.array(m20)[np.array(m20) > 0]
        Y_grid_filterd = np.array(self.Y_grid)[np.array(m20) > 0]
        Y_dist_filtered = m20_filtered * Y_grid_filterd
        
        # Calculate the Lorenz curve
        Lorenz_Y20 = np.cumsum(Y_dist_filtered)/np.sum(Y_dist_filtered)
        Lorenz_Y20 = np.insert(Lorenz_Y20, 0 , 0.0)
        
        self.Lorenz_Y20 = Lorenz_Y20
        
        # Plot the Lorenz curve        
        x2 = np.cumsum(m20_filtered)/np.sum(m20_filtered)
        x2 = np.insert(x2, 0 , 0.0)
        
        self.x2 = x2
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(x, x, '-', linewidth = 0.5, color = 'green', label='perfect equality')
        plt.plot(x, Lorenz_Y20_original, color='blue', ls='dashed',
                 label='Lorenz curve computed in (b)')
        plt.plot(x2, Lorenz_Y20, '-r', lw = 3, label='Lorenz curve for Y_20')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig3.savefig('Lorenz_Y20.png', dpi=300)



            
        