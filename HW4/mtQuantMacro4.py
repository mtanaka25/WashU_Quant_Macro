#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW4.py

is the python class for the assignment #4 of Quantitative Macroeconomic Theory
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

class IFP: # Income fluctuation problem
    def __init__(self,
                 beta      = 0.9400, # discount factor
                 sig       = 2.0000, # inverse IES 
                 R         = 1.0100, # gross interest rate
                 a_lb      = 0.0000, # borrowing limit
                 rho       = 0.9900, # AR coefficient of income process
                 var_eps   = 0.0426, # variance of income shock
                 n_grids_y = 17, # # of gird points for y
                 n_grids_a = 500, # # of gird points for a 
                 a_range   = (0, 5), # range of a interval # TODO! Choose this value wisely.  
                 Omega     = 3.0000, # half of interval range (for tauchen)
                 ):
        # calculate sigma_y based on rho and var_eps
        sig_eps = np.sqrt(var_eps)
        sig_lny = sig_eps * (1 - rho**2)**(-1/2)
        var_lny = sig_lny**2
        
        # prepare the grid points for a
        a_grid = np.linspace(a_range[0], a_range[1], n_grids_a)
        
        # Store the given parameters as instance attributes
        self.beta      = beta
        self.sig       = sig
        self.R         = R
        self.a_lb      = a_lb
        self.rho       = rho
        self.var_eps   = var_eps
        self.sig_eps   = sig_eps
        self.var_lny   = var_lny
        self.sig_lny   = sig_lny
        self.n_grids_y = n_grids_y
        self.n_grids_a = n_grids_a
        self.a_grid    = a_grid
        self.Omega     = Omega
    
    
    def discretize(self, method,
                is_write_out_discretization_result = True,
                is_quiet = False): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            if not is_quiet:
                print("\n Discretizing the income process by Tauchen method")
            self.tauchen_discretize(is_write_out_discretization_result)
        
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("\n Discretizing the income process by Rouwenhorst method")
            self.rouwenhorst_discretize(is_write_out_discretization_result)
            
        else:
            raise Exception('"method" input much be "Tauchen" or "Rouwenhorst."')
    
    
    def tauchen_discretize(self, is_write_out_discretization_result):
        # Prepare gird points
        lny_N    = self.Omega * self.sig_lny
        lny_grid = np.linspace(-lny_N, lny_N, self.n_grids_y)
        y_grid   = np.exp(lny_grid)
        
        # Calculate the step size
        h = (2 * lny_N)/(self.n_grids_y-1)
        
        # Construct the transition matrix
        trans_mat = [ 
            [self.tauchen_trans_mat_ij(i, j, lny_grid, h) 
             for j in range(self.n_grids_y)
            ]
            for i in range(self.n_grids_y)
            ]
            
        if is_write_out_discretization_result:
            np.savetxt('Tauchen_y_grid.csv', y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Tauchen_trans_mat.csv', trans_mat, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
        
        self.lny_grid, self.y_grid, self.trans_mat, self.step_size =\
            lny_grid, y_grid, np.array(trans_mat), h
    
    
    def tauchen_trans_mat_ij(self, i, j, lny_grid, h):
        if j == 0:
            trans_mat_ij = norm.cdf((lny_grid[j] - self.rho*lny_grid[i] + h/2)/self.sig_eps)
        elif j == (self.n_grids_y-1):
            trans_mat_ij = 1 - norm.cdf((lny_grid[j] - self.rho*lny_grid[i] - h/2)/self.sig_eps)
        else:
            trans_mat_ij = ( norm.cdf((lny_grid[j] - self.rho*lny_grid[i] + h/2)/self.sig_eps)
                           - norm.cdf((lny_grid[j] - self.rho*lny_grid[i] - h/2)/self.sig_eps))
        return trans_mat_ij
    
    
    def rouwenhorst_discretize(self, is_write_out_discretization_result):
        # Prepare y gird points
        lny_N    = self.sig_lny * np.sqrt(self.n_grids_y - 1)
        lny_grid = np.linspace(-lny_N, lny_N, self.n_grids_y)
        y_grid   = np.exp(lny_grid)
        
        # Calculate the step size
        h = (2 * lny_N)/(self.n_grids_y-1)
        
        # parameter necessary for Rouwenhorst recursion
        pi = 0.5 * (1 + self.rho)
        
        # N = 2
        Pi_N = np.array([[pi, 1 - pi],
                         [1 - pi, pi]])
        
        for n in range(3, self.n_grids_y+1, 1):
            Pi_pre = deepcopy(Pi_N)
            Pi_N1, Pi_N2, Pi_N3, Pi_N4 = \
                np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
            
            Pi_N1[:n-1, :n-1] = Pi_N2[:n-1, 1:n] = \
                Pi_N3[1:n, 1:n] = Pi_N4[1:n, :n-1] = Pi_pre
            
            Pi_N = (pi * Pi_N1
                    + (1 - pi) * Pi_N2
                    + pi * Pi_N3
                    + (1 - pi) * Pi_N4
            )
            # Divide all but the top and bottom rows by two so that the 
            # elements in each row sum to one (Kopecky & Suen[2010, RED]).
            Pi_N[1:-1, :] *= 0.5
            
        if is_write_out_discretization_result:
            np.savetxt('Rouwenhorst_y_grid.csv', y_grid, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Rouwenhorst_trans_mat.csv', Pi_N, delimiter=' & ', 
                       fmt='%2.3f', newline=' \\\\\n')
            
        self.lny_grid, self.y_grid, self.trans_mat, self.step_size \
            = lny_grid, y_grid, Pi_N, h
    
    
    def utility(self, a_td, y_td, a_tmrw):
        # Calculate consumption as the residual in the budget constraint
        c_td = self.R*a_td + y_td - a_tmrw
        # If consumption is negative, assign NaN
        c_td[c_td < 0] = np.nan
        # Compute the instantaneous utility
        u_td = c_td**(1-self.sig) / (1 - self.sig)
        
        return u_td
    
    
    def muc(self, a_td, y_td, a_tmrw): # marginal utility of consumption
        # Calculate consumption as the residual in the budget constraint
        c_td = self.R*a_td + y_td - a_tmrw
        # If consumption is negative, assign NaN
        c_td[c_td <= 0] = np.nan
        # Compute the marginal utility of consumption
        muc_td = c_td**(-self.sig) 
        
        return muc_td
    
    def repmat_2dto3d(self, mat, n_rep):
        array_3d = [mat for i in range(n_rep)]
        array_3d = np.array(array_3d)
        array_3d = np.transpose(array_3d, (1, 2, 0))     
        return array_3d
    
    def dot_3d(self, A, B):
        AB = [A[:,:,i] @ B for i in range(A.shape[2])]
        AB = np.array(AB)
        AB = np.transpose(AB, (1, 2, 0))
        return AB
    
    def Euler_eq_resid(self, a_td_idx, y_td_idx, a_tmrw_idx_mat):
    # Today's state variables (scalers)
        a_td, y_td = self.a_grid[a_td_idx], self.y_grid[y_td_idx]
        
        # Optimal asset holdings tommorrow (scaler)
        a_tmrw_idx = a_tmrw_idx_mat[a_td_idx, y_td_idx]
        a_tmrw = self.a_grid[a_tmrw_idx]
        
        # Possible asset holdings day after tomorrow (vector)
        a_dat_idx_vec = a_tmrw_idx_mat[a_tmrw_idx, :]
        a_dat_vec = self.a_grid[a_dat_idx_vec]
        
        # Left-hand side of the Euler equation (scaler)
        LHS = self.muc(a_td = a_td,
                       y_td = y_td,
                       a_tmrw = a_tmrw)
        
        # Right-hand side of the Euler equation
        muc_tmrw_vec = self.muc(a_td = self.repmat_2dto3d(a_tmrw, self.n_grids_y),
                                y_td = (self.y_grid).reshape(1, -1),
                                a_tmrw = a_dat_vec) # vector
        RHS = self.beta * self.R * np.sum(self.dot_3d(self.trans_mat, muc_tmrw_vec), axis=2)
        
        # The difference between LHS and RHS (EE = Euler equation)
        EE_resid = LHS - RHS
        
        return EE_resid
    
    
    def find_optimal_a_tmrw(self, a_td_idx, y_td_idx, a_tmrw_idx_mat):
        # Today's state variables
        a_td, y_td = self.a_grid[a_td_idx], self.y_grid[y_td_idx] # scalers
        
        # Possible asset holdings tommorrow (horizontal vector)
        a_tmrw_idx_vec = np.arange(self.n_grids_a).reshape(1, -1)
        a_tmrw_vec = (self.a_grid).reshape(1, -1)
        
        # Possible asset holdings day after tomorrow (matrix)
        a_dat_idx_mat = (a_tmrw_idx_mat.T)[a_tmrw_idx_vec, :]
        a_dat_mat = self.a_grid[a_dat_idx_mat]
        
        # Possbile values of left-hand side of the Euler equation
        LHS = self.muc(a_td = a_td,
                       y_td = y_td,
                       a_tmrw = a_tmrw_vec) # horizontal vector
        
        # Possbile values of right-hand side of the Euler equation
        muc_tmrw_mat = self.muc(a_td = a_tmrw_vec,  
                                y_td = (self.y_grid).reshape(-1,1),
                                a_tmrw = a_dat_mat) 
        RHS = self.beta * self.R * np.sum(self.dot_3d(muc_tmrw_vec, self.trans_mat), axis=2)
        
        # The difference between LHS and RHS (EE = Euler equation)
        EE_resid_vec = LHS - RHS
        
        # find the value for a' most consistent with the Euler equation
        # and return its index
        # Note: Monotonicity of marginal utility is exploited.
        abs_EE_resid_vec = abs(EE_resid_vec)
        optimal_a_idx = abs_EE_resid_vec.index(np.nanmin(abs_EE_resid_vec))
        
        return optimal_a_idx
    
    
    def policy_function_iter(self,
                             tol = 0,
                             max_iter = 10000):
        a_idx_vec = np.arange(self.n_grids_a).reshape(-1, 1)
        y_idx_vec = np.arange(self.n_grids_y).reshape(1, -1)
        
        # Initial guess for the policy function (Use the 45 degree line)
        a_hat_idx_mat = np.tile(a_idx_vec, (1, self.n_grids_y))
        self.a_hat_idx_mat = a_hat_idx_mat
        
        # Initialize while loop
        is_converged, iteration = False, 0
        
        while (not is_converged) & (iteration <= max_iter):
            # Prepare a guess for the policy function
            a_hat_idx_guess_mat = deepcopy(a_hat_idx_mat)
            
            # Check if the borrowing constraint is binding
            EE_resid = self.Euler_eq_resid(a_td_idx  = a_idx_vec,
                                           y_td_idx  = y_idx_vec,
                                           a_tmrw_idx_mat = a_hat_idx_guess_mat) # matrix
            is_a_lb_binding = (EE_resid > 0) # If Ture, the constraint is binding
            
            # Construct an updated policy function matrix
            a_hat_idx_mat = [
                [0 if is_a_lb_binding[i,j]==True 
                  else self.find_optimal_a_tmrw[i,j, a_hat_idx_guess_mat]
                  for i in range(self.n_grids_a)]
                for j in range(self.n_grids_y)]
            a_hat_idx_mat = np.array(a_hat_idx_mat)
            
            diff = np.sum(a_hat_idx_guess_mat == a_hat_idx_mat)
            is_converged = (diff <= tol)
            iteration += 1
            
        self.a_hat_idx_mat
            
            

