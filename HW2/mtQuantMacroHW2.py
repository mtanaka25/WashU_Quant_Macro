#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW2.py

is the python class for the assignment #2 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 14, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn
from tabulate import tabulate
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

class GHHModel:
    def __init__(self,
                 alpha  = 0.330, # capital share in production function
                 beta   = 0.960, # discount factor
                 theta  = 1.000, # inverse Frisch elasticity
                 gamma  = 2.000, # risk avrsion
                 omega  = 2.000, # convexity of the depreciation func.
                 B      = 0.075, # coefficient in the depreciation func.
                 A      = 0.592, # TFP level or list for TFP development
                 sigma  = 0.060, # s.d. of the stochastic process
                 lmbd   = 0.400, # Autocorrelation of the stochastic process
                 nGrids = 100,   # # of grid points for k
                 k_min  = 0.000, # the lower bound of k
                 k_max  = 5.000  # the upper bound of k
                 ):
        k_grid = np.linspace(k_min, k_max, nGrids)
        if k_min == 0:
            # to avoid the degenerated steady state, do not use k = 0.
            k_grid[0] = 0.001 * k_grid[1]
        elif k_min < 0:
            raise Exception('The lower bound of k is expected greater than zero.')
        
        # list of the possible shock values
        eps_list = [sigma, -sigma]       
        # Construct the transition matrix for the shock process
        pi11   = 0.5 * (lmbd + 1)
        pi12   = 1 - pi11
        pi22   = 0.5 * (lmbd + 1)
        pi21   = 1 - pi22
        prob = np.array([[pi11, pi21],
                         [pi12, pi22]])      
                
        # Store the parameter values as instance attributes
        self.alpha  = alpha
        self.beta   = beta
        self.theta  = theta
        self.gamma  = gamma
        self.omega  = omega
        self.B      = B
        self.A      = A
        self.nGrids = nGrids
        self.k_grid = k_grid
        self.prob   = prob
        self.eps_list = eps_list        
        
    def find_optimal_h_l(self, 
                        k, # capital stock today 
                        eps, # shock realized today
                        h_min =  0.0, # lower bound of utilixation rate
                        h_max =  1.0, # upper bound of utilixation rate 
                                      # (Note h is in [0,1])
                        l_min =  0.0, # lower bound of labor input
                        l_max = 50.0  # upper bound of labor input
                        ):
        # TODO! We can obtain the closed solutions of h and l. Rewrite this method with them.
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
        # Convert the return of argmax (int) into a coordinate array (tuple of ints)
        idx0, idx1 = np.unravel_index(np.argmax(obj_func_val, axis = None), 
                                      obj_func_val.shape)
        
        h_hat = h_grid[idx0, 0] # Pick up the optimal utilization rate
        l_hat = l_grid[0, idx1] # Pick up the optimal labor input
        
        return h_hat, l_hat
    
    
    def eval_value_func(self, 
                        k_idx, # index for the capital stock today 
                        eps_idx, # index for the shock realized today
                        V_tmrw, # tomorrow's values (depending on k')
                        is_concave = False, # if true, exploit concavity
                        starting_grid = 0,
                        ):
        """
        eval_value_func evaluates today's value based on the given
        capital stock today (k), the given investment-specific shock today
        (epsilon), and tomorrow's value (V'). Note that V' depends on the
        capital stock tomorrow (k'), which is optimally chosen today.
        """
        # load necessary parameter values from instance attributes
        alpha  = self.alpha
        beta   = self.beta
        theta  = self.theta
        omega  = self.omega
        gamma  = self.gamma
        A      = self.A
        B      = self.B
        nGrids = self.nGrids
        k_grid = self.k_grid
        prob_r = self.prob[:, eps_idx]
        k      = self.k_grid[k_idx]
        eps    = self.eps_list[eps_idx]
        
        # Allocate memory for the vector of possible today's value
        possible_V_td = np.empty((nGrids, ))
        possible_V_td[:]= np.nan
        
        # Find the optimal utilization rate and the optimal
        # labor input under the given k and epsilon
        h_hat, l_hat = self.find_optimal_h_l(k_idx, eps_idx)
        
        count=0
        for i in range(int(starting_grid), nGrids, 1):
            # capital stock tomorrow
            k_tmrw = k_grid[i]
            
            # Calculate the optimal consumption depending on k',
            c_i = (
                A * (k*h_hat)**alpha * l_hat**(1-alpha)
                - k_tmrw * np.exp(-eps)
                + k * (1 - B * h_hat**omega / omega) * np.exp(-eps)
                )
        
            # Calculate the inside of brackets in the utility function
            u_i = c_i - (l_hat**(1+theta)) / (1+theta)

                    
            # Check if the inside is positive
            # Otherwise, cannot compute powers for some gamma values
            #   e.g. (-0.5)**(0.2) is not defined in real numbers
            if u_i > 0:
                count+=1
                # calculate corresponding value if the inside is positive
                # Otherwise, do nothing (leave the value at NaN)
                u_i = 1/(1-gamma) * u_i**(1-gamma)
                possible_V_td[i] = u_i + beta * np.dot(V_tmrw[i, :], prob_r)
                
            if is_concave and (i > 0):
                if possible_V_td[i - 1] > possible_V_td[i]:
                # If the possible value decreases from that in the previous 
                # loop (and concavity is satisfied), stop calculating V
                    break
                            
        # take the maximum in the possible today's value vector
        self.count = count
        self.u_i = u_i
        self.V_tmrw = V_tmrw
        self.possible_V_td = possible_V_td
        self.k_idx = k_idx
        self.eps_idx= eps_idx
        self.i= i
        self.starting_grid = starting_grid
        k_tmrw_idx = np.nanargmax(possible_V_td)
        V_td_k_eps = possible_V_td[k_tmrw_idx]   
                
        return V_td_k_eps, k_tmrw_idx
    
    
    def value_func_iter(self,
                        V_init   = np.nan,
                        tol      = 10E-5,
                        max_iter = 1000,
                        is_concave = False,  # if true, exploit concavity
                        is_monotone = False, # if true, exploit monotonicity
                        is_modified_policy_iter = False, # if true, implement modified policy itereation
                        n_h = 10,
                        ):
        # if initial guess for V is not given, start with zero matrix
        if np.isnan(V_init):
            V_init = np.zeros((self.nGrids, len(self.eps_list)))
        
        # Initialize while-loop
        i      = 0
        diff   = 1.0
        V_post = V_init.copy()
        policy_func = np.zeros((self.nGrids, len(self.eps_list)))
        
        
        # Start stopwatch
        tic = time.time()
        
        # Value function iteration
        while (i < max_iter) and (diff > tol):
            V_pre = V_post.copy()
            
            # Value function iteration part
            for r in range(len(self.eps_list)):
                starting_grid = 0
                for j in range(self.nGrids):            
                    V_post[j, r], policy_func[j, r] = self.eval_value_func(j, r, V_pre, is_concave, starting_grid)
                    if is_monotone:
                        starting_grid = policy_func[j, r]
            diff = np.nanmax((np.abs(V_post - V_pre)))
            
            
            # Modified policy function iteration part
            if is_modified_policy_iter:
                if diff <= tol:
                    # if already converged, exit without policy iteration
                    break
                V_post = self.modified_policy_func_iter(
                    V_init = V_post,
                    policy_func = policy_func,
                    n_h = n_h
                )
                diff = np.nanmax(np.abs(V_post - V_pre))
            
            # Proceed the iteration counter
            i+=1 
        
        # Check if max_iter is binding
        if diff > tol:
            raise Exception('Value function iteration reached max_iter. The solution could be incorrect.')
        
        # Stop stopwatch
        toc = time.time()
        elapsed_time = toc - tic
        
        # Save result as instance attributes
        self.elapsed_time = elapsed_time
        self.V            = V_post
        self.policy_fuc   = policy_func
    
    
    def modified_policy_func_iter(
                                self, 
                                V_init, # tomorrow's value prior to policy iteration
                                policy_func, # policy function obtained in the previous step
                                n_h, # # of iterations
                                ):
        alpha    = self.alpha
        beta     = self.beta
        theta    = self.theta
        omega    = self.omega
        gamma    = self.gamma
        A        = self.A
        B        = self.B
        nGrids   = self.nGrids
        k_grid   = self.k_grid
        prob_r   = self.prob
        eps_list = self.eps_list
        
        V_post = V_init.copy()
        
        for i in range(n_h):
            V_pre = V_post.copy()
            for j in range(nGrids):
                k = k_grid[j]
                for r in range(len(eps_list)):
                    eps = eps_list[r]
                    
                    # "optimal" capital stock
                    k_tmrw_idx = int(policy_func[j, r])
                    k_tmrw = k_grid[k_tmrw_idx]
                    
                    # Optimal utilization rate and labor
                    h_hat, l_hat = self.find_optimal_h_l(k, eps)
                    
                    # optimal consumption
                    c = (
                        A * (k*h_hat)**alpha * l_hat**(1-alpha)
                        - k_tmrw * np.exp(-eps)
                        + k * (1 - B * h_hat**omega / omega) * np.exp(-eps)
                        )
                    
                    # the value inside the brackets in utility function
                    u = c - (l_hat**(1+theta)) / (1+theta)
                    
                    if u <= 0:
                        V_post[j, r] = np.nan
                    else:
                        u = 1/(1-gamma) * u**(1-gamma)
                        V_post[j, r] = u + beta * np.dot(V_pre[k_tmrw_idx, :], prob_r[:, r])
        return V_post



class DataStatsGenerator:
    def __init__(self,
                 api_key,
                 ticker_Y='',
                 ticker_C='',
                 ticker_I='',
                 ticker_L='',
                 ticker_A='',
                 ticker_H=''
                 ):
        self.api_key  = api_key
        self.ticker_Y = ticker_Y
        self.ticker_C = ticker_C
        self.ticker_I = ticker_I
        self.ticker_L = ticker_L
        self.ticker_A = ticker_A
        self.ticker_H = ticker_H
    
    def obtain_data(self):
        # Connect to FRED
        fred = Fred(api_key = self.api_key)
        
        # Obtain the necessary data from FRED
        self.Y_obs = self.obtain_individual_data(fred, self.ticker_Y)
        self.C_obs = self.obtain_individual_data(fred, self.ticker_C)
        self.I_obs = self.obtain_individual_data(fred, self.ticker_I)
        self.L_obs = self.obtain_individual_data(fred, self.ticker_L)
        self.A_obs = self.obtain_individual_data(fred, self.ticker_A)
        self.H_obs = self.obtain_individual_data(fred, self.ticker_H)
    
    def obtain_individual_data(self, fred, ticker):
        if ticker == '':
            ln_data_cyc = np.NaN
        else:
            data = fred.get_series(self.ticker)
            if self.isMonthly(data):
                data = data.resample('Q').mean()
            ln_data = np.log(data)
            ln_data_cyc, _ = hpfilter(ln_data.dropna(), 1600)
        return ln_data_cyc
    
    def isMonthly(self, data):
        date_series = pd.Series(data.index)
        date_series = date_series.dt.to_period("Q")
        if (date_series[0] == date_series[1]) or (date_series[1] == date_series[2]):
            isMonthly = True
        else:
            isMonthly = False
        return isMonthly
    
    def calc_stats(self):
        self.Y_std, self.Y_corr, self.Y_auto = self.calc_individual_stats(self.Y_obs)
        self.C_std, self.C_corr, self.C_auto = self.calc_individual_stats(self.C_obs)
        self.I_std, self.I_corr, self.I_auto = self.calc_individual_stats(self.I_obs)
        self.L_std, self.L_corr, self.L_auto = self.calc_individual_stats(self.L_obs)
        self.A_std, self.A_corr, self.A_auto = self.calc_individual_stats(self.A_obs)
        self.H_std, self.H_corr, self.H_auto = self.calc_individual_stats(self.H_obs)
    
    def calc_individual_stats(self, data):
        if np.isnan(data):
            std, corr, auto = np.NaN, np.NaN, np.NaN
        else:
            std  = data.std()
            corr = data.corr(self.Y_obs)
            auto = data.autocorr()
        return std, corr, auto
