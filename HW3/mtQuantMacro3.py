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
import time
from copy import deepcopy
from scipy.stats import norm, lognorm
from scipy.optimize import minimize


class SIMModel_super:
    def __init__(self, 
                 age_range = (20, 65)
                 ):
        # Prepare list of ages
        age_list = [i for i in range(age_range[0], age_range[1]+1, 1)]
        
        # Store the list as an instance attribute
        self.age_list = age_list
    
    
    def calc_Lorenz_curve(self, Y_vec, distribution_vec):
        if type(Y_vec) is list:
            Y_vec = np.array(Y_vec)
        elif type(Y_vec) is pd.core.series.Series:
            Y_vec = Y_vec.to_numpy()
        
        if type(distribution_vec) is list:
            distribution_vec= np.array(distribution_vec)
        elif type(distribution_vec) is pd.core.series.Series:
            distribution_vec = distribution_vec.to_numpy()
        
        # Calculate the cumulative share in aggregate earnings
        Y_contrib = Y_vec * distribution_vec
        cum_Y_share = np.cumsum(Y_contrib)/np.sum(Y_contrib)
        cum_Y_share = np.insert(cum_Y_share, 0 , 0.0)
        
        # Calculate the cumulative share in total samples
        cum_N_share = np.cumsum(distribution_vec)/np.sum(distribution_vec)
        cum_N_share = np.insert(cum_N_share, 0 , 0.0)
        
        Lorenz_curve = np.array([cum_N_share, cum_Y_share])
        return Lorenz_curve
    
    
    def calc_Gini_index(self, Lorenz_curve):
        Gini_contrib = [(Lorenz_curve[1, i] + Lorenz_curve[1, i+1])
                        *(Lorenz_curve[0, i+1] - Lorenz_curve[0, i])
                        *0.5
                        for i in range(np.size(Lorenz_curve, 1) - 1)]
        Gini_index = (0.5 -  np.sum(Gini_contrib)) / 0.5
        
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
        # Run the super class's __init__
        super().__init__(age_range)
        
        # Store the given parameters as instance attributes
        self.rho       = rho
        self.var_eps   = var_eps
        self.sig_eps   = np.sqrt(var_eps)
        self.var_y20   = var_y20
        self.sig_y20   = np.sqrt(var_y20)
        self.n_grids   = n_grids
        self.Omega     = Omega
        self.n_samples = n_samples
        
    def discretize(self, method,
                is_write_out_discretization_result = True,
                is_quiet = False): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            if not is_quiet:
                print("\n Discretizing the AR(1) process by Tauchen method")
            self.tauchen_discretize(is_write_out_discretization_result)
        
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("\n Discretizing the AR(1) process by Rouwenhorst method")
            self.rouwenhorst_discretize(is_write_out_discretization_result)
            
        else:
            raise Exception('"method" input much be "Tauchen" or "Rouwenhorst."')
    
    
    def tauchen_discretize(self, is_write_out_discretization_result):
        # Prepare y gird points
        y_N     = self.Omega * self.sig_y20
        y_grid  = np.linspace(-y_N, y_N, self.n_grids)
        Y_grid  = np.exp(y_grid)
        
        # Calculate the step size
        h = (2 * y_N)/(self.n_grids-1)
        
        # Construct the transition matrix
        trans_mat = [ 
            [self.tauchen_trans_mat_ij(i, j, y_grid, h) 
             for j in range(self.n_grids)
            ]
            for i in range(self.n_grids)
            ]
            
        if is_write_out_discretization_result:
            np.savetxt('Tauchen_Y_grid.csv', Y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Tauchen_trans_mat.csv', trans_mat, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
        
        self.Y_grid = Y_grid
        self.y_grid = y_grid
        self.trans_mat = np.array(trans_mat)
        self.step_size = h
    
    
    def tauchen_trans_mat_ij(self, i, j, y_grid, h):
        if j == 0:
            trans_mat_ij = norm.cdf((y_grid[j] - self.rho*y_grid[i] + h/2)/self.sig_eps)
        elif j == (self.n_grids-1):
            trans_mat_ij = 1 - norm.cdf((y_grid[j] - self.rho*y_grid[i] - h/2)/self.sig_eps)
        else:
            trans_mat_ij = ( norm.cdf((y_grid[j] - self.rho*y_grid[i] + h/2)/self.sig_eps)
                           - norm.cdf((y_grid[j] - self.rho*y_grid[i] - h/2)/self.sig_eps))
        return trans_mat_ij
    
    
    def rouwenhorst_discretize(self, is_write_out_discretization_result):
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
            
            Pi_N1[:n-1, :n-1] = Pi_pre
            Pi_N2[:n-1, 1:n] = Pi_pre
            Pi_N3[1:n, 1:n] = Pi_pre
            Pi_N4[1:n, :n-1] = Pi_pre
            
            Pi_N = (pi * Pi_N1
                    + (1 - pi) * Pi_N2
                    + pi * Pi_N3
                    + (1 - pi) * Pi_N4
            )
            # Divide all but the top and bottom rows by two so that the 
            # elements in each row sum to one (Kopecky & Suen[2010, RED]).
            Pi_N[1:-1, :] *= 0.5
            
        if is_write_out_discretization_result:
            np.savetxt('Rouwenhorst_Y_grid.csv', Y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Rouwenhorst_trans_mat.csv', Pi_N, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            
        self.Y_grid = Y_grid
        self.y_grid = y_grid
        self.trans_mat = Pi_N
        self.step_size = h
        
        return y_grid, Pi_N
    
    
    def run_simulation(self, fname_header='model', fixed_seed=None, is_plot=True):
        if fixed_seed != None:
            np.random.seed(fixed_seed)
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Draw samples
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        Y20_samples = lognorm.rvs(self.sig_y20, size=self.n_samples)
        Y20_samples = np.sort(Y20_samples)
                
        if is_plot:
            fig0 = plt.figure(figsize=(8, 6))
            plt.hist(Y20_samples, bins=25, density=True)
            
            fig0.savefig(fname_header+'_Sample_histgram.png', 
                         dpi=300, bbox_inches='tight', pad_inches=0)
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Derive and plot Lorenz curve for earnings itself (not log earning)
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Calculate the Lorenz curve
        cum_Y20_share = np.cumsum(Y20_samples)/np.sum(Y20_samples)
        cum_N_share =  np.linspace(0, 1, self.n_samples)
        
        Lorenz_Y20_original = np.array([cum_N_share, cum_Y20_share])
        
        self.Lorenz_Y20_original = Lorenz_Y20_original
        
        if is_plot:
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
            fig1.savefig(fname_header+'_Lorenz_Y20_original.png', 
                        dpi=300, bbox_inches='tight', pad_inches=0)
        
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
        self.m20 = m20
        
        if is_plot:
            fig2 = plt.figure(figsize=(8, 6))
            plt.bar(self.Y_grid, m20)
            fig2.savefig(fname_header+'_PDF_Y20.png',
                        dpi=300, bbox_inches='tight', pad_inches=0)      
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Lorenz curve for Y20 based on grid points 
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        Lorenz_Y20 = self.calc_Lorenz_curve(self.Y_grid, m20)
        self.Lorenz_Y20 = Lorenz_Y20
        
        if is_plot:
            fig3 = plt.figure(figsize=(8, 6))
            plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[0],
                     '-', linewidth = 0.5, color = 'green', label='perfect equality')
            plt.plot(Lorenz_Y20[0], Lorenz_Y20[1], 
                     '-r', lw = 3, label='Lorenz curve for Y_20')
            plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[1],
                     color='blue', ls='dashed', label='Lorenz curve computed in (b)')
            plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                       linestyles='--', label='perfect inequality')
            plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                       linestyles='--')
            plt.legend(frameon = False)
            fig3.savefig(fname_header+'_Lorenz_Y20.png',
                        dpi=300, bbox_inches='tight', pad_inches=0)
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Lorenz curves for 3 age groups
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
        Lorenz_Y20_25 = self.calc_Lorenz_curve(self.Y_grid, y_20_25)
        
        # group for ages 40-45
        y_40_45 = np.sum(m_mat[20:25, :], axis=0)
        Lorenz_Y40_45 = self.calc_Lorenz_curve(self.Y_grid, y_40_45)
        
        # group for ages 60-65
        y_60_65 = np.sum(m_mat[40:45, :], axis=0)
        Lorenz_Y60_65 = self.calc_Lorenz_curve(self.Y_grid, y_60_65)
        
        if is_plot:
            fig4 = plt.figure(figsize=(8, 6))
            plt.plot(Lorenz_Y20_original[0], Lorenz_Y20_original[0],
                     '-', linewidth = 0.5, color = 'green', 
                     label='perfect equality')
            plt.plot(Lorenz_Y60_65[0], Lorenz_Y60_65[1],
                     color='r', lw=3, label='Ages 60-65')
            plt.plot(Lorenz_Y40_45[0], Lorenz_Y40_45[1], 
                     'purple', label='ages 40-45')
            plt.plot(Lorenz_Y20_25[0], Lorenz_Y20_25[1],
                     color='b', ls='dashed', label='Ages 20-25')
            plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                       linestyles='--', label='perfect inequality')
            plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                       linestyles='--')
            plt.legend(frameon = False)
            fig4.savefig(fname_header+'_Lorenz_groups.png',
                        dpi=300, bbox_inches='tight', pad_inches=0)
        
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Compute the Gini coefficient for each age
        # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+           
        
        Lorenz_by_age = [self.calc_Lorenz_curve(self.Y_grid, m_mat[a, :])
                         for a in range(n_ages)]
        Gini_by_age =[self.calc_Gini_index(Lorenz_by_age[a])
                      for a in range(n_ages)]
        
        self.Lorenz_by_age = Lorenz_by_age
        self.Gini_by_age = Gini_by_age
        
        if is_plot:
            fig5 = plt.figure(figsize = (8, 6))
            plt.plot(self.age_list, Gini_by_age, color='r')
            fig5.savefig(fname_header+'_Gini_coefficients_by_age.png',
                        dpi=300, bbox_inches='tight', pad_inches=0)



class SCF_hetero_income(SIMModel_super):
    def __init__(self, 
                 fname, # The name of SCF data file 
                 age_range = (20, 65),
                 ):
        # Run the super class's __init__\
        super().__init__(age_range)
        
        # Load SCF data
        df = pd.read_stata(fname)
        
        # Pick up the necessary items
        # -- 'wgt': weight, 'age': age, 'wageinc': wage income 
        df = df.filter(['wgt', 'age', 'wageinc']) 
        
        # Wage income for some samples is recorded as zero.
        # Exclude such samples.
        df = df.replace(0, np.nan)
        df = df.dropna()
        
        # Exclude samples whose age is not in the sample ages
        df = df[(df.age>(self.age_list[0]-1))&(df.age<(self.age_list[-1]+1))]
        
        df = df.sort_values('wageinc')
        self.df = df
    
    def calc_Lorenz_for_three_groups(self, fname_header='SCF'):
        Lorenz_Y20_25 = self.calc_Lorenz_curve(
            (self.df[self.df.age<26]).wageinc,
            (self.df[self.df.age<26]).wgt
            )
        Lorenz_Y40_45 = self.calc_Lorenz_curve(
            (self.df[(self.df.age>39)&(self.df.age<46)]).wageinc,
            (self.df[(self.df.age>39)&(self.df.age<46)]).wgt
            )
        Lorenz_Y60_65 = self.calc_Lorenz_curve(
            (self.df[self.df.age>59]).wageinc,
            (self.df[self.df.age>59]).wgt
            )
        
        fig0 = plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0,1,5), np.linspace(0,1,5),
                 '-', linewidth = 0.5, color = 'green', 
                 label='perfect equality')
        plt.plot(Lorenz_Y20_25[0], Lorenz_Y20_25[1],
                 color='b', ls='dashed', label='Ages 20-25')
        plt.plot(Lorenz_Y40_45[0], Lorenz_Y40_45[1], 'purple', 
                 label='ages 40-45')
        plt.plot(Lorenz_Y60_65[0], Lorenz_Y60_65[1],
                 color='r', lw=3, label='Ages 60-65')
        plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--', label='perfect inequality')
        plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
                   linestyles='--')
        plt.legend(frameon = False)
        fig0.savefig(fname_header+'_Lorenz_by_group.png',
                    dpi=300, bbox_inches='tight', pad_inches=0)
        
        
    def calc_Gini_index_by_age(self, fname_header='SCF'):
        Lorenz_by_age = [self.calc_Lorenz_curve(
                        (self.df[self.df.age==age]).wageinc,
                        (self.df[self.df.age==age]).wgt
                        )
                         for age in self.age_list]
        Gini_by_age =[self.calc_Gini_index(Lorenz_by_age[i])
                      for i in range(len(Lorenz_by_age))]
        
        self.Lorenz_by_age = Lorenz_by_age
        self.Gini_by_age = Gini_by_age
        
        fig = plt.figure(figsize = (8, 6))
        plt.plot(self.age_list, Gini_by_age, color='r')
        fig.savefig(fname_header+'_Gini_coefficients_by_age.png',
                    dpi=300, bbox_inches='tight', pad_inches=0)
    
    
    def recalibrate_AR_params(self, param0):
        # This is a trick to restrict rho in [0,1]
        param0[0] = param0[0]/(1 - param0[0])
        print('\n Recalibrating the parameters...')
        print('\n This process will take much time. Please be patient.')
        tic = time.time()
        minimize_result = minimize(self.diff_obsGini_TauchenGini, param0,
                                   method='Nelder-Mead', tol=1e-2,
                                   options={'maxiter': 10**5, 'maxfev': 10**5,
                                            'disp': True}
                                   )
        self.minimize_result = minimize_result
        toc = time.time()
        self.elapsed_time = toc - tic
        print('\n Finished recalibrating!')
        
        optimal_param = minimize_result.x
        optimal_param[0] = abs(optimal_param[0])/(1 + abs(optimal_param[0]))
        return optimal_param
    
    
    def diff_obsGini_TauchenGini(self, param0):
        # Below is the trick to estimate rho in [0,1]
        rho     = abs(param0[0])/(1 + abs(param0[0]))
        var_eps = param0[1]**2
        var_y20 = param0[2]**2
        
        model = SIMModel(rho     = rho,
                         var_eps = var_eps,
                         var_y20 = var_y20)
        model.discretize(method='tauchen', 
                         is_write_out_discretization_result = False,
                         is_quiet=True)
        model.run_simulation(is_plot=False)
        
        diff = [abs(self.Gini_by_age[i] - model.Gini_by_age[i])
                for i in range(5, 41, 1)]
        diff = np.sum(diff)
        return diff
        