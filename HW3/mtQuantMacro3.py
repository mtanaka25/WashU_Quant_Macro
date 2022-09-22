#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW3.py

is the python class for the assignment #3 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 20, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
from http.cookies import SimpleCookie
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from scipy.stats import norm, lognorm
from copy import deepcopy
from tabulate import tabulate

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults


class SIMModel_super:
    def __init__(self):
        pass
    
    def discrete_Gini_index(self, Lorenz_curve):
        Gini_contrib = [(Lorenz_curve[1, i-1] + Lorenz_curve[1, i])*(Lorenz_curve[0, i] - Lorenz_curve[0, i-1])*0.5
                        for i in range(1, np.size(Lorenz_curve, 1))]
        Gini_contrib.insert(0, Lorenz_curve[1, 1] *Lorenz_curve[0, 1] * 0.5)
        
        Gini_index = 0.5 -  np.sum(Gini_contrib)
        
        return Gini_index
        

class SIMModel(SIMModel_super):
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
        
    def discretize(self, method,
                is_save_y_grid = True,
                is_save_trans_mat = True): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            print("\n Dscretizing the AR(1) process by Tauchen method")
            self.tauchen_discretize(is_save_y_grid, is_save_trans_mat)
        
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            print("\n Dscretizing the AR(1) process by Rouwenhorst method")
            self.rouwenhorst_discretize(is_save_y_grid, is_save_trans_mat)
            
        else:
            raise Exception('"method" input much be "Tauchen" or "Rouwenhorst."')
        
        
    def tauchen_discretize(self, is_save_y_grid, is_save_trans_mat):
        # Prepare y gird points
        y_N     = self.Omega * self.sig_y20
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
        self.trans_mat = np.array(trans_mat)
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
        
        
    def rouwenhorst_discretize(self, is_save_y_grid, is_save_trans_mat):
        # Prepare y gird points
        y_N     = self.sig_y20 * np.sqrt(self.n_grids - 1)
        y_grid  = np.linspace(-y_N, y_N, self.n_grids)
        Y_grid  = np.exp(y_grid)
        
        # Calculate the step size
        h = (2 * y_N)/(self.n_grids-1)
        
        # parameter necessary for Rouwenhorst recursion
        pi = 0.5 * (1 + self.rho)
        
        # N = 2
        Pi_N = np.array([[pi, 1 - pi],
                         [1 - pi, pi]])
        
        for n in range(3, self.n_grids+1, 1):
            Pi_pre = deepcopy(Pi_N)
            Pi_N1  = np.zeros((n,n))
            Pi_N2  = np.zeros((n,n))
            Pi_N3  = np.zeros((n,n))
            Pi_N4  = np.zeros((n,n))
            
            Pi_N1[0:n-1, 0:n-1] = Pi_pre
            Pi_N2[0:n-1, 1:n] = Pi_pre
            Pi_N3[1:n, 1:n] = Pi_pre
            Pi_N4[1:n, 0:n-1] = Pi_pre
            
            Pi_N = (pi * Pi_N1
                    + (1 - pi) * Pi_N2
                    + pi * Pi_N3
                    + (1 - pi) * Pi_N4
            )
            
            # Divide all but the top and bottom rows by two so that the 
            # elements in each row sum to one (Kopecky & Suen[2010, RED]).
            Pi_N[1:-2, :] *= 0.5
        
        self.Y_grid = Y_grid
        self.y_grid = y_grid
        self.trans_mat = Pi_N
        self.step_size = h
        
        return y_grid, Pi_N
    
    
    def run_simulation(self, fname_header='model', fixed_seed=None):
        if fixed_seed != None:
            np.random.seed(fixed_seed)
        
        Y20_samples = lognorm.rvs(self.sig_y20, size=self.n_samples)
        Y20_samples = np.sort(Y20_samples)
        
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Derive and plot Lorenz curve for earnings itself (not log earning)
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        fig0 = plt.figure(figsize=(8, 6))
        plt.hist(Y20_samples, bins=25, density=True)
        fig0.savefig(fname_header+'_Sample_histgram.png', dpi=300)
        
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Derive and plot Lorenz curve for earnings itself (not log earning)
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Calculate the Lorenz curve
        cum_Y20_share = np.cumsum(Y20_samples)/np.sum(Y20_samples)
        cum_N_share =  np.linspace(0, 1, self.n_samples)
        
        Lorenz_Y20_original = np.array([cum_N_share, cum_Y20_share])
        
        self.Lorenz_Y20_original = Lorenz_Y20_original
        
        
        # Plot the Lorenz curve        
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[0], 
                 '-', lw = 0.5, color = 'green', label='perfect equality')
        plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[1],
                 '-r', lw = 3, label='Lorenz curve for Y_20')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig1.savefig(fname_header+'_Lorenz_Y20_original.png', dpi=300)
        
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
        
        fig2.savefig(fname_header+'_PDF_Y20.png', dpi=300)
        
        self.m20 = m20
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Lorenz curve based on grid points 
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        Lorenz_Y20 = self.discrete_Lorenz_curve(m20)
        self.Lorenz_Y20 = Lorenz_Y20
        
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[0],
                 '-', linewidth = 0.5, color = 'green', label='perfect equality')
        plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[1],
                 color='blue', ls='dashed', label='Lorenz curve computed in (b)')
        plt.plot(Lorenz_Y20[0], Lorenz_Y20[1], '-r', lw = 3, label='Lorenz curve for Y_20')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig3.savefig(fname_header+'_Lorenz_Y20.png', dpi=300)
        
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Lorenz curve based on grid points 
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+       
        n_ages = len(self.age_list)
        m_mat = np.empty((n_ages, self.n_grids))
        
        m_mat[0, :] = m20
        
        m_a = np.array(m20).reshape(1, -1)
        
        for a in range(n_ages-1):
            m_a_1 = m_a @ self.trans_mat
            m_mat[a+1, :] = m_a_1.flatten()
            m_a = deepcopy(m_a_1)
        self.m_mat = m_mat
        
        # group for ages 20-25
        y_20_25 = np.sum(m_mat[0:5, :], axis=0)
        Lorenz_Y20_25 = self.discrete_Lorenz_curve(y_20_25)
        
        # group for ages 40-45
        y_40_45 = np.sum(m_mat[20:25, :], axis=0)
        Lorenz_Y40_45 = self.discrete_Lorenz_curve(y_40_45)
        
        # group for ages 60-65
        y_60_65 = np.sum(m_mat[40:45, :], axis=0)
        Lorenz_Y60_65 = self.discrete_Lorenz_curve(y_60_65)
        
        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[0],
                 '-', linewidth = 0.5, color = 'green', label='perfect equality')
        plt.plot(Lorenz_Y20_25[0], Lorenz_Y20_25[1],
                 color='blue', ls='dashed', label='Ages 20-25')
        plt.plot(Lorenz_Y40_45[0], Lorenz_Y40_45[1], 'purple', label='ages 40-45')
        plt.plot(Lorenz_Y60_65[0], Lorenz_Y60_65[1],
                 color='red', lw=3, label='Ages 60-65')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig4.savefig(fname_header+'_Lorenz_groups.png', dpi=300)
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Gini coefficient for each age
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+           
        
        Lorenz_by_age = [self.discrete_Lorenz_curve(m_mat[a, :])
                         for a in range(n_ages)]
        self.Lorenz_by_age = Lorenz_by_age
        Gini_by_age =[self.discrete_Gini_index(Lorenz_by_age[a])
                      for a in range(n_ages)]
        self.Gini_by_age = Gini_by_age
        
        fig5 = plt.figure(figsize = (8, 6))
        plt.plot(self.age_list, Gini_by_age, color='red', lw = 3)
        fig5.savefig(fname_header+'_Gini_coefficients_by_age.png', dpi=300)
        
    def discrete_Lorenz_curve(self, distribution):
        if type(distribution) is list:
            distribution = np.array(distribution)
        
        dist_filtered = distribution[distribution > 0]
        Y_grid_filterd = np.array(self.Y_grid)[distribution > 0]
        
        # Calculate the cumulative share in aggregate earnings
        Y_contrib_filtered = dist_filtered * Y_grid_filterd
        cum_Y_share = np.cumsum(Y_contrib_filtered)/np.sum(Y_contrib_filtered)
        cum_Y_share = np.insert(cum_Y_share, 0 , 0.0)
        
        # Calculate the cumulative share in total samples
        cum_N_share = np.cumsum(dist_filtered)/np.sum(dist_filtered)
        cum_N_share = np.insert(cum_N_share, 0 , 0.0)
        
        Lorenz_curve = np.array([cum_N_share, cum_Y_share])
        return Lorenz_curve


class SCF_hetero_income(SIMModel_super):
    def __init__(self, fname):
        # Load SCF data
        df = pd.read_stata(fname)
        
        # Pick up the necessary items
        # -- 'wgt': weight, 'age': age, 'wageinc': wage income 
        df = df.filter(['wgt', 'age', 'wageinc']) 
        
        # Wage income for some samples is recorded as zero.
        # Exclude such samples.
        df = df.replace(0, np.nan)
        df = df.dropna()
        
        # Exclude samples whose age is not in [20, 65]
        df = df[(df.age>19)&(df.age<66)]
        
        df = df.sort_values('wageinc')
        self.df = df
    
    def empirical_Lorenz_curve(self, Y_vec, weight_vec):
        if type(Y_vec) is list:
            Y_vec = np.array(Y_vec)
        elif type(Y_vec) is pd.core.series.Series:
            Y_vec = Y_vec.to_numpy()
        
        if type(weight_vec) is list:
            weight_vec = np.array(weight_vec)
        elif type(weight_vec) is pd.core.series.Series:
            weight_vec = weight_vec.to_numpy()
        
        # Calculate the cumulative share in aggregate earnings
        Y_contrib = Y_vec * weight_vec
        cum_Y_share = np.cumsum(Y_contrib)/np.sum(Y_contrib)
        cum_Y_share = np.insert(cum_Y_share, 0 , 0.0)
        
        # Calculate the cumulative share in total samples
        cum_N_share = np.cumsum(weight_vec)/np.sum(weight_vec)
        cum_N_share = np.insert(cum_N_share, 0 , 0.0)
        
        Lorenz_curve = np.array([cum_N_share, cum_Y_share])
        return Lorenz_curve
    
    def calc_Lorenz_for_three_groups(self, fname_header='SCF'):
        
        Lorenz_Y20_25 = self.empirical_Lorenz_curve(
            (self.df[self.df.age<26]).wageinc,
            (self.df[self.df.age<26]).wgt
            )
        Lorenz_Y40_45 = self.empirical_Lorenz_curve(
            (self.df[(self.df.age>39)&(self.df.age<46)]).wageinc,
            (self.df[(self.df.age>39)&(self.df.age<46)]).wgt
            )
        
        Lorenz_Y60_65 = self.empirical_Lorenz_curve(
            (self.df[self.df.age>59]).wageinc,
            (self.df[self.df.age>59]).wgt
            )
        
        fig0 = plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0,1,5), np.linspace(0,1,5),
                 '-', linewidth = 0.5, color = 'green', label='perfect equality')
        plt.plot(Lorenz_Y20_25[0], Lorenz_Y20_25[1],
                 color='blue', ls='dashed', label='Ages 20-25')
        plt.plot(Lorenz_Y40_45[0], Lorenz_Y40_45[1], 'purple', label='ages 40-45')
        plt.plot(Lorenz_Y60_65[0], Lorenz_Y60_65[1],
                 color='red', lw=3, label='Ages 60-65')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig0.savefig(fname_header+'_Lorenz_by_group.png', dpi=300)        
        
        
    def calc_Gini_index_by_age(self, fname_header='SCF'):
        
        Lorenz_by_age = [self.empirical_Lorenz_curve(
                        (self.df[self.df.age==age]).wageinc,
                        (self.df[self.df.age==age]).wgt
                        )
                         for age in range(20, 66, 1)]
        self.Lorenz_by_age = Lorenz_by_age
        Gini_by_age =[self.discrete_Gini_index(Lorenz_by_age[i])
                      for i in range(len(Lorenz_by_age))]
        self.Gini_by_age = Gini_by_age
        
        fig = plt.figure(figsize = (8, 6))
        plt.plot(range(20, 66, 1), Gini_by_age, color='red', lw = 3)
        fig.savefig(fname_header+'_Gini_coefficients_by_age.png', dpi=300)     