#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW6.py

is the python class for the assignment #6 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Oct 19, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import numpy as np
from numpy.random import seed, uniform
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy

# Load the personal Python package, which is based on the assignments #1-#5.
from mtPyEcon import AR1_process, PiecewiseIntrpl_MeshGrid
from mtPyTools import StopWatch, find_nearest_idx

class KV2010:
    # ======== The methods in this section are not expected directly called in main.py ==========
    # ======== These would work in the background ===============================================
    def __init__(self,
                 a_lb      = 0.,  # borrowing constraint
                 a_max     = 30., # max
                 age_range = (25, 80), # range of ages
                 first_old_age = 66, # the age when the old-age period begins
                 N_a       = 500,  # # of grid points for a
                 beta      = 0.950, # subjective discount factor
                 R         = 1.010, # Gross interest rate
                 sig       = 2.000, # risk aversion
                 rho       = 0.995, # AR coefficient in persistent shock
                 var_eta   = 0.010, # variance of persistent shock
                 var_z0    = 0.150, # initial variance of z
                 var_eps   = 0.050, # variance of epsilon
                 mu_eps    = 0.000, # mean of epsilon
                 penalty   = 1E-20, # penalty consumption if c is negative
                 ):
        
        # Prepare grid points for a
        a_grid = np.linspace(a_lb, a_max, N_a)
        
        # Prepare the vector of ages
        age_vec = np.arange(age_range[0], age_range[1]+1)
        old_age_vec = np.arange(first_old_age, age_range[1]+1)
        working_age_vec = np.arange(age_range[0], first_old_age)
        
        # Define AR(1) process using the AR1_process class
        z_process = AR1_process(rho = rho,
                                sig = np.sqrt(var_eta),
                                varname = 'z',
                                sig_init_val = np.sqrt(var_z0),
                                is_stationary = False)
        
        # discretize epsilon
        eps_vec = np.array([mu_eps - np.sqrt(var_eps/2), mu_eps + np.sqrt(var_eps/2)])
        
        # Store the model information as instance attributes
        self.beta, self.R, self.sig = beta, R, sig
        self.eps_vec = eps_vec
        self.z_process = z_process
        self.N_a, self.N_eps = N_a, 2
        self.N_age, self.N_w_age = len(age_vec), len(working_age_vec)
        self.age_vec = age_vec
        self.old_age_vec = old_age_vec
        self.working_age_vec = working_age_vec
        self.a_grid = a_grid
        self.penalty = penalty

    
    def age_idx(self, age):
        age_idx = age - self.age_vec[0]
        return int(age_idx)
    
    def discretize_z_process(self,
                             method = 'Rouwenhorst',
                             N_z = 20, # number of grid points
                             Omega = 3, # scale parameter for Tauchen grid
                             is_write_out_result = False,
                             is_quiet = False):
        self.z_process.discretize(method = method,
                                  N = N_z,
                                  Omega = Omega,
                                  approx_horizon = len(self.working_age_vec),
                                  is_write_out_result = is_write_out_result,
                                  is_quiet = is_quiet)
        z_grid_list = self.z_process.z_grid_list
        trans_mat_list = self.z_process.trans_mat_list
        trans_mat_list.append(np.eye(N_z))
        for t in range(self.N_age - self.N_w_age):
            z_grid_list.append(z_grid_list[-1])
            trans_mat_list.append(trans_mat_list[-1])
            
        # For future convienience, take out grid and transition matrix from AR1 instance.
        self.discretization_method = method
        self.z_grid_list = z_grid_list
        self.trans_mat_list = trans_mat_list
        self.N_z = N_z
    
    
    def utility(self, age, a, a_prime, z, eps = 0):
        # income at the given age
        Y = self.income(age, z, eps)
        # Calculate consumption from the budget constraint
        c = self.R * a + Y - a_prime
        # If c is negstive, give a penalty
        if (np.isscalar(c)):
            if c <= 0:
                c = self.penalty
        else:
            c[c<=0] = self.penalty
        # Calculate the CES utility
        u = c**(1-self.sig) / (1-self.sig)
        return u
    
    def income(self, age, z, eps):
        if np.isscalar(age):
            if age > self.working_age_vec[-1]:
                Y = self.pension(z)
            else:
                Y = self.earnings(age, z, eps)
        else:
            Y = np.zeros((len(age),))
            for i, age_i in enumerate(age):
                if age_i > self.working_age_vec[-1]:
                    Y[i] = self.pension(z[i])
                else:
                    Y[i] = self.earnings(age_i, z[i], eps[i])
        return Y
    
    def deterministic_earnings(self,
                                age,
                                coef0 = -1.0135,
                                coef1 =  0.1086,
                                coef2 = -0.001122):
        det_earnings = np.log(coef0 + coef1*age + coef2 * age**2)
        return det_earnings
    
    def earnings(self, age, z, eps):
        lnY = self.deterministic_earnings(age) + z + eps
        Y = np.exp(lnY)
        return Y
    
    def pension(self, z):
        pension_flow = 0.7 * np.exp(z + self.deterministic_earnings(self.working_age_vec[-1]))
        return pension_flow
    
    def make_income_array(self):
        income_array = [[[
            [self.income(age, z, eps) for a in self.a_grid]
            for z in self.z_grid_list[i]]
            for eps in self.eps_vec]
            for i, age in enumerate(self.age_vec)]
        self.income_array = np.array(income_array)
    
    def solve_for_V(self, a, z, eps, age, V_prime, interpolate=False):
        # if the sample reaches the last age of life, the value of the state is
        # simply the instantaneous utility. And, in that case, the sample is
        # no longer allowrd to borrow.
        if age == self.age_vec[-1]:
            V = self.utility(a = a, z = z, eps = eps, age = age, a_prime = 0)
            a_prime_star = 0
        else:
            # pick up z_grid for the given age
            z_grid = self.z_grid_list[self.age_idx(age)]
            # the current z's index in z grid
            z_idx = find_nearest_idx(z, z_grid)
            # pick up the transition matrix for the given age
            if interpolate:
                trans_mat = self.trans_mat_finer_list[self.age_idx(age)]
            else:
                trans_mat = self.trans_mat_list[self.age_idx(age)]
            # prepare the (horizontal) vector of a'
            if interpolate:
                a_prime = (self.a_finer_grid).reshape(1, -1)
            else:
                a_prime = (self.a_grid).reshape(1, -1)
            # Calculate the possible values
            utility = self.utility(a=a, z=z, eps=eps, age=age, a_prime=a_prime)
            expected_V = trans_mat[z_idx, :] @ [0.5 * V_prime[0, :, :] + 0.5 * V_prime[1, :, :]]
            possible_V = utility + self.beta * expected_V
            # Take max
            V = np.nanmax(possible_V)
            a_prime_star_idx = np.nanargmax(possible_V)
            a_prime_star = a_prime[0, a_prime_star_idx]
        # Return the maximized value and the optimal asset holding
        return V, a_prime_star
    
    
    def value_func_iter(self, interpolate=False, N_z_finer=100, N_a_finer=2000):
        # Prepare the terminal value of V' (The afterlife is supposed worthless.)
        V_prime = np.zeros((self.N_eps, self.N_z, self.N_a))
        
        # Prepare 4D arrays where the matrices of V and a' for each age and epsilon will be stored
        V_4Darray = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        a_prime_4Darray = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        
        # When using interpolation, do necessary preparations
        if interpolate:
            # back up the original transition matrix and grid points
            z_grid_original, trans_mat_original = deepcopy(self.z_grid_list), deepcopy(self.trans_mat_list)
            
            # Rerun discretization to obtain the finer transition matrix
            # -- Be careful that self.z_gird and self.trans_mat would be overwritten
            self.discretize_z_process(method = self.discretization_method,
                                      N_z = N_z_finer,
                                      is_write_out_result = False,
                                      is_quiet = True
                                      )
            
            # exchange the variable names
            self.z_grid_list, self.z_finer_grid_list = z_grid_original, self.z_grid_list
            
            # Reduce the size of the transition matrix to N_A * N_A_finer
            for i in range(len(self.trans_mat_list)):
                z_grid_correspondence = find_nearest_idx(self.z_grid_list[i], self.z_finer_grid_list[i])
                self.trans_mat_list[i] = self.trans_mat_list[i][z_grid_correspondence, :]
            
            # exchange the variable names
            self.trans_mat_list, self.trans_mat_finer_list = trans_mat_original, self.trans_mat_list
            
            # Prepare the finer grid for a
            self.a_finer_grid = np.linspace(self.a_grid[0], self.a_grid[-1], N_a_finer)
        
        # start stop watch
        stopwatch = StopWatch()
        print('Solving backward the discretized model...\n')
        # Solve backward (with respect to age)
        for age in reversed(self.age_vec):
            age_idx = self.age_idx(age)
            z_grid = self.z_grid_list[age_idx]
            if interpolate: # If use interpolation, do so.
                z_finer_grid = self.z_finer_grid_list[age_idx]
                V_prime_0_intrpl = PiecewiseIntrpl_MeshGrid(z_grid, self.a_grid, V_prime[0, :, :])
                V_prime_1_intrpl = PiecewiseIntrpl_MeshGrid(z_grid, self.a_grid, V_prime[1, :, :])
                if age < self.working_age_vec[-1]:
                    V_prime_0_finer = V_prime_0_intrpl(z_finer_grid, self.a_finer_grid)
                    V_prime_1_finer = V_prime_1_intrpl(z_finer_grid, self.a_finer_grid)
                else:
                    # After retiring, z does not change. So, interpolate only wrt a.
                    V_prime_0_finer = V_prime_0_intrpl(z_grid, self.a_finer_grid)
                    V_prime_1_finer = V_prime_1_intrpl(z_grid, self.a_finer_grid)
                V_prime = np.array([V_prime_0_finer, V_prime_1_finer])
            
            # calculate the maximized V and the optimal b' for each epsilon, z and a
            for eps_idx, eps in enumerate(self.eps_vec):
                for z_idx, z in enumerate(z_grid):
                    for a_idx, a in enumerate(self.a_grid):
                        V_for_this_state, a_prime_for_this_state =\
                            self.solve_for_V(a = a,
                                             z = z,
                                             eps = eps,
                                             age = age,
                                             V_prime = V_prime,
                                             interpolate = interpolate)
                        V_4Darray[age_idx, eps_idx, z_idx, a_idx] = deepcopy(V_for_this_state)
                        a_prime_4Darray[age_idx, eps_idx, z_idx, a_idx] = deepcopy(a_prime_for_this_state)
            # Use the calculated V as V' in the next loop
            V_prime = deepcopy(V_4Darray[age_idx, :, :, :])
        stopwatch.stop()
        #Store the solution
        self.V = V_4Darray
        self.a_prime = a_prime_4Darray
    
    
    def get_stochastic_path(self, init_z_idx, init_eps_idx):
        # Prepare the nested functions
        cumsum_transmat_list = [
            np.cumsum(self.trans_mat_list[i], axis = 1)
            for i in range(len(self.trans_mat_list))
            ]
        def draw_z_prime_idx(z_idx, age_idx):
            # Prepare criteria to pin down z'
            cumsum_transmat_z = cumsum_transmat_list[age_idx][z_idx, :]
            # Draw a random number
            rand_val = uniform(0, 1)
            # Decide z' depending on the random draw
            z_prime_idx = np.sum(cumsum_transmat_z < rand_val)
            return int(z_prime_idx)
        def draw_eps_prime_idx():
            # Choose the epsilon' idx which is closer to random draw
            return int(round(uniform(0, 1)))
        
        # Prepare the index vectors for sample z and epsilon
        # (The vectors contain indices, so dtype is a kind of integer)
        eps_path_idx    = np.zeros(self.age_vec.shape, dtype = np.uint8)
        eps_path_idx[0] = init_eps_idx
        z_path_idx   = np.zeros(self.age_vec.shape, dtype = np.uint8)
        z_path_idx[0]= init_z_idx
        
        for i in range(1, self.N_age, 1):
            if i < len(self.working_age_vec):
                # If the age (i) is in working age, draw z' and epsilon'
                eps_path_idx[i] = draw_eps_prime_idx()
                z_path_idx[i]   = draw_z_prime_idx(z_path_idx[i-1], age_idx = i-1)
            else:
                # If the age is in old age, fix z at the level when retiring
                # Note that while epsilon is no longer irrelevant to old ages,
                # we repeat epsilon as well in order to simplify the script
                eps_path_idx[i] = eps_path_idx[i-1]
                z_path_idx[i] = z_path_idx[i-1]
        return eps_path_idx, z_path_idx
    
    
    def simulate_single_sample(self,
                                init_a = 0, # initial asset holding
                                init_z_idx = 4, # the index of z at age 25
                                init_eps_idx = 0 # the index of epsilon at age 25
                                ):
        # Draw random epsion and z
        eps_path_idx, z_path_idx = \
            self.get_stochastic_path(init_z_idx = init_z_idx,
                                     init_eps_idx = init_eps_idx)
        # Simulate the subject's life using the drawn shocks
        a_prime_path = np.zeros(self.age_vec.shape)
        c_path = np.zeros(self.age_vec.shape)
        Y_path = np.zeros(self.age_vec.shape)
        # initial value of a
        a =  deepcopy(init_a)
        a_idx = find_nearest_idx(a, self.a_grid)
        for i, age_i in enumerate(self.age_vec):
            eps_i = self.eps_vec[eps_path_idx[i]]
            z_i = self.z_grid_list[i][z_path_idx[i]]
            Y_path[i] = self.income(age_i, z_i, eps_i)
            a_prime_i = self.a_prime[i,
                                    eps_path_idx[i],
                                    z_path_idx[i],
                                    a_idx]
            c_path[i] = self.R * a + Y_path[i] - a_prime_i
            a_prime_path[i] = a_prime_i
            # Set a_prime today to a tomorrow
            a = deepcopy(a_prime_i)
            a_idx = find_nearest_idx(a, self.a_grid)
        return Y_path, a_prime_path, c_path
    
    
    def monte_carlo_simulation(self,
                               init_a,
                               init_z_idx,
                               init_eps_idx,
                               N_samples = 25_000,
                               ):
        # Prepare matrices for simulation result
        Y_path_mat = np.zeros((N_samples, self.N_age))
        c_path_mat = np.zeros((N_samples, self.N_age))
        a_prime_path_mat = np.zeros((N_samples, self.N_age))
        
        print('Running simulation with {0} samples...\n'.format(N_samples))
        stopwatch = StopWatch()
        for i in range(N_samples):
            Y_path_i, a_prime_path_i, c_path_i = \
                self.simulate_single_sample(init_a = init_a,
                                            init_z_idx   = init_z_idx,
                                            init_eps_idx = init_eps_idx)
            Y_path_mat[i, :] = Y_path_i
            c_path_mat[i, :] = c_path_i
            a_prime_path_mat[i, :] = a_prime_path_i
            if i % 1_000 == 0:
                print('Sample {0}: Done...'.format(i))
        stopwatch.stop()
        
        # Take sample average
        Y_path_mean = np.mean(Y_path_mat, axis = 0)
        c_path_mean = np.mean(c_path_mat, axis = 0)
        a_prime_path_mean = np.mean(a_prime_path_mat, axis = 0)
        
        # Calc variance of log income and log consumption
        lnY_var_path = np.var(np.log(Y_path_mat), axis = 0)
        lnc_var_path = np.var(np.log(c_path_mat), axis = 0)
        
        # Store the simulation result as instance attribute
        self.Y_path_mat, self.c_path_mat, self.a_prime_path_mat = \
            Y_path_mat, c_path_mat, a_prime_path_mat
        self.Y_path_mean, self.c_path_mean, self.a_prime_path_mean = \
            Y_path_mean, c_path_mean, a_prime_path_mean
        self.lnY_var_path, self.lnc_var_path = \
            lnY_var_path, lnc_var_path
    
    
    def calc_insurance_coef(self):
        # to calculate the coefficient for working age, the path during
        # old age would be cut off
        end_idx = int(len(self.working_age_vec))
        # convert sample data into log-differences from the previous age
        dY_mat = (np.log(self.Y_path_mat[1:, :end_idx])
                  - np.log(self.Y_path_mat[:-1, :end_idx])).flatten()
        dc_mat = (np.log(self.c_path_mat[1:, :end_idx])
                  - np.log(self.c_path_mat[:-1, :end_idx])).flatten()
        # calculate variance and covariace matrix of deltaY and deltac
        varcov_mat_dY_dc = np.cov(dY_mat, dc_mat)
        # calculate the coefficent
        insurance_coef = 1 - varcov_mat_dY_dc[0, 1]/varcov_mat_dY_dc[0, 0]
        # Store the coef in the instance
        self.insurance_coef = insurance_coef
    
    
    def solve_for_distribution(self):
        # Prepare a 4D array for the initial distribution
        # -- The distribution is given a 4D array whose size is
        #        (# of age) *  (# of possible epsilon) * (# of possible z) * (# of possible a)
        population = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        # The distribution at age 25 is assumed as follows.
        # -- Nobady is assumed to hold a strictly positive amount of asset at the starting age
        # -- z and epsilon are uniformly distributed
        no_asset_idx = find_nearest_idx(0, self.a_grid)
        population[0, :, :, no_asset_idx] = 1 / (self.N_eps * self.N_z)
        
        print('Solving for the distribution...\n')
        stopwatch = StopWatch()
        for age_idx in range(len(self.age_vec) - 1):
            population[age_idx + 1, :, :, :] = \
                self._solve_for_dist_routine(age_idx, population[age_idx, :, :, :])
        population = population/np.sum(population)
        stopwatch.stop()
        self.distribution = population
    
    def _solve_for_dist_routine(self, age_idx, pdf_pre):
        # Simplify notation
        a = self.a_grid
        a_prime = self.a_prime
        # Pick up the transition matrix
        trans_mat = self.trans_mat_list[age_idx]
        # Define a nested function to update each entry in pdf
        def updated_pdf_element(z_idx, a_idx):
            def indicator(a_array, idx):
                condition_1 = np.zeros(a_array.shape)
                condition_2 = np.zeros(a_array.shape)
                if idx == 1:
                    condition_1[a[idx-1] <= a_array] = 1
                    condition_2[a[idx] >= a_array] = 1
                else:
                    condition_1[a[idx-1] < a_array] = 1
                    condition_2[a[idx] >= a_array] = 1
                indicator_mat = condition_1 * condition_2
                return indicator_mat
            
            if  a_idx == 0:
                below_from_epsL, below_from_epsH = 0, 0
            else:
                below_from_epsL = np.sum(
                        0.5 * trans_mat.T[z_idx, :] @ (
                        indicator(a_prime[age_idx, 0, :, :], a_idx)
                        * (a_prime[age_idx, 0, :, :] - a[a_idx-1])/(a[a_idx] - a[a_idx-1])
                        * pdf_pre[0, : , :]
                        ))
                below_from_epsH = np.sum(
                        0.5 * trans_mat.T[z_idx, :] @ (
                        indicator(a_prime[age_idx, 1, :, :], a_idx)
                        * (a_prime[age_idx, 1, :, :] - a[a_idx-1])/(a[a_idx] - a[a_idx-1])
                        * pdf_pre[1, : , :]
                        ))
            if  a_idx == len(self.a_grid)-1:
                above_from_epsL, above_from_epsH = 0, 0
            else:
                above_from_epsL = np.sum(
                        0.5 * trans_mat.T[z_idx, :] @ (
                        indicator(a_prime[age_idx, 0, :, :], a_idx+1)
                        * (a[a_idx+1] - a_prime[age_idx, 0, :, :])/(a[a_idx+1] - a[a_idx])
                        * pdf_pre[0, : , :]
                        ))
                above_from_epsH = np.sum(
                        0.5 * trans_mat.T[z_idx, :] @ (
                        indicator(a_prime[age_idx, 1, :, :], a_idx+1)
                        * (a[a_idx+1] - a_prime[age_idx, 1, :, :])/(a[a_idx+1] - a[a_idx])
                        * pdf_pre[1, : , :]
                        ))
            element = below_from_epsL + below_from_epsH + above_from_epsL + above_from_epsH
            return element
        
        # Update the pdf
        pdf_at_the_age = [[
            [updated_pdf_element(z_idx, a_idx) for a_idx in range(self.N_a)]
            for z_idx in range(self.N_z)]
            for eps_idx in range(self.N_eps)]
        
        return np.array(pdf_at_the_age)
    
    
    def calc_aggregate_asset(self):
        agg_asset = np.sum(self.distribution * self.a_prime)
        self.aggregate_asset = agg_asset
        
    # ======== The following methods are expected directly called in main.py ==========
    def solve_question_1a(self,
                        method = 'Rouwenhorst',
                        ages2plot = (25, 40, 60),
                        z2plot    = (4, 9, 14),
                        interpolate = False,
                        N_z_finer = 100,
                        N_a_finer = 2000,
                        fname = 'Q1(a).png'
                        ):
        # solve the model
        self.discretize_z_process(method = method)
        self.value_func_iter(interpolate = interpolate,
                            N_z_finer   = N_z_finer,
                            N_a_finer   = N_a_finer)
        
        # graphics
        ages_idx = find_nearest_idx(ages2plot, self.age_vec)
        
        fig, ax = plt.subplots(3, 1, figsize=(12, 16))
        
        for i in range(2):
            ax[i].plot(self.a_grid,self.a_grid,
                        lw = 0.75, c = 'black', label = '45 degree line')
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[0], 0, z2plot[i], :].flatten(),
                        lw = 1.5, c = 'gray', ls = 'dashed',
                        label='{0} years old'.format(ages2plot[0]))
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[1], 0, z2plot[i], :].flatten(),
                        lw = 1.5, c = 'blue',
                        label='{0} years old'.format(ages2plot[1]))
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[2], 0, z2plot[i], :].flatten(),
                        lw = 2.5, c = 'red',
                        label='{0} years old'.format(ages2plot[2]))
            ax[i].set_xlabel("$a$")
            ax[i].set_title("$a'(a | z{0}, \\varepsilon_L, age)$".format([5, 10][i]))
            ax[i].legend(frameon=False)
        ax[2].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', label = '45 degree line')
        ax[2].plot(self.a_grid, self.a_prime[ages_idx[1], 0, z2plot[2], :].flatten(),
                        lw = 1.5, c = 'blue',
                        label='\\varepsilon_L')
        ax[2].plot(self.a_grid, self.a_prime[ages_idx[1], 1, z2plot[2], :].flatten(),
                        lw = 2.5, c = 'red',
                        label='\\varepsilon_H')
        ax[2].set_xlabel("$a$")
        ax[2].set_title("$a'(a | z15, \\varepsilon, 40)$")
        ax[2].legend(frameon=False)
        plt.savefig(fname, dpi = 150, bbox_inches='tight', pad_inches=0)
    
    
    def solve_question_1b(self,
                          init_a = 0.,
                          init_z_idx = 4,
                          init_eps_idx = 0,
                          fix_seed = None,
                          fname = 'Q1(b).png'
                          ):
        if not(type(fix_seed) is type(None)):
            seed(fix_seed)
        
        Y_path, a_prime_path, c_path = self.simulate_single_sample(init_a = init_a,
                                                              init_z_idx   = init_z_idx,
                                                              init_eps_idx = init_eps_idx)
        
        fig, ax = plt.subplots(1, 1,  figsize=(12, 8))
        x_label = self.age_vec
        ax.plot(x_label, c_path,
                c = 'green',lw = 1.5, label = '$c$')
        ax.plot(x_label, a_prime_path,
                c = 'red',  lw = 1.5, label = "$a'$")
        ax.plot(x_label, Y_path,
                c = 'blue', lw = 1.5, label = '$Y$')
        ax.set_xlabel("age")
        ax.legend(frameon=False)
        plt.savefig(fname, dpi = 150, bbox_inches='tight', pad_inches=0)
    
    
    def solve_question_1c(self,
                            init_a = 0.,
                            init_z_idx = 5,
                            init_eps_idx = 0,
                            N_samples = 25_000,
                            fix_seed = None,
                            fname = 'Q1(c).png'
                            ):
        # Set seed if necessary
        if not(type(fix_seed) is type(None)):
            seed = fix_seed
        
        self.monte_carlo_simulation(init_a = init_a,
                                    init_z_idx = init_z_idx,
                                    init_eps_idx = init_eps_idx,
                                    N_samples = N_samples,
                                    )
        
        self.calc_insurance_coef()
        
        # Graphics
        fig, ax = plt.subplots(2, 1,  figsize=(12, 8))
        x_label = self.age_vec
        ax[0].plot(x_label, self.c_path_mean,
                   c = 'green', lw = 1.5, label = '$c$')
        ax[0].plot(x_label, self.a_prime_path_mean,
                   c = 'red',   lw = 1.5, label = "$a'$")
        ax[0].plot(x_label, self.Y_path_mean,
                   c = 'blue',  lw = 1.5, label = '$Y$')
        ax[0].set_xlabel("age")
        ax[0].legend(frameon=False)
        
        ax[1].plot(x_label, self.lnc_var_path,
                   c = 'green', lw = 1.5, label = '$var(ln c)$')
        ax[1].plot(x_label, self.lnY_var_path,
                   c = 'blue',  lw = 1.5, label = '$var(ln Y)$')
        ax[1].set_xlabel("age")
        ax[1].legend(frameon=False)
        plt.savefig(fname, dpi = 150, bbox_inches='tight', pad_inches=0)
    
    def solve_question_2a(self,
                          bins  = 12,
                          a_max = 7,
                          y_max = 7,
                          fnames = ('Q2(a)1.png', 'Q2(a)2.png'),
                          ):
        if not hasattr(self, "distribution"):
            self.solve_for_distribution()
        if not hasattr(self, "income_array"):       
            self.make_income_array()
        
        # **************
        fig1, ax1 = plt.subplots(2, 3,  figsize=(12, 8))
        ax1[0, 0].hist(x = self.income_array[ 0, :, : ,:].flatten(),
                       weights = self.distribution[ 0, :, : ,:].flatten()/np.sum(self.distribution[ 0, :, : ,:]),
                       range = (0, y_max),
                       bins = bins)
        ax1[0, 0].set_title('income, age 25')
        ax1[0, 1].hist(x = self.income_array[15, :, : ,:].flatten(),
                       weights = self.distribution[15, :, : ,:].flatten()/np.sum(self.distribution[15, :, : ,:]),
                       range = (0, y_max),
                       bins = bins)
        ax1[0, 1].set_title('income, age 40')
        ax1[0, 2].hist(x = self.income_array[35, :, : ,:].flatten(),
                       weights = self.distribution[35, :, : ,:].flatten()/np.sum(self.distribution[35, :, : ,:]),
                       range = (0, y_max),
                       bins = bins)
        ax1[0, 2].set_title('income, age 60')
        ax1[1, 0].hist(x = self.a_prime[ 0, :, : ,:].flatten(),
                       weights = self.distribution[ 0, :, : ,:].flatten()/np.sum(self.distribution[ 0, :, : ,:]),
                       range = (self.a_grid[0], a_max),
                       bins = bins)
        ax1[1, 0].set_title('asset, age 25')
        ax1[1, 1].hist(x = self.a_prime[15, :, : ,:].flatten(),
                       weights = self.distribution[15, :, : ,:].flatten()/np.sum(self.distribution[15, :, : ,:]),
                       range = (self.a_grid[0], a_max),
                       bins = bins)
        ax1[1, 1].set_title('asset, age 40')
        ax1[1, 2].hist(x = self.a_prime[35, :, : ,:].flatten(),
                       weights = self.distribution[35, :, : ,:].flatten()/np.sum(self.distribution[35, :, : ,:]),
                       range = (self.a_grid[0], a_max),
                       bins = bins)
        ax1[1, 1].set_title('asset, age 60')
        plt.savefig(fnames[0], dpi = 150, bbox_inches='tight', pad_inches=0)
        
        # **************
        fig2, ax2 = plt.subplots(2, 1,  figsize=(12, 8))
        ax2[0].hist(x = self.income_array.flatten(),
                       weights = self.distribution.flatten(),
                       range = (0, y_max),
                       bins = bins)
        ax2[0].set_title('income, all ages')
        ax2[1].hist(x = self.a_prime.flatten(),
                       weights = self.distribution.flatten(),
                       range = (self.a_grid[0], a_max),
                       bins = bins)
        ax2[1].set_title('asset, all ages')
        plt.savefig(fnames[1], dpi = 150, bbox_inches='tight', pad_inches=0)
        
    def solve_question_2b(self, is_quiet = False):
        self.calc_aggregate_asset()
        if not is_quiet:
            print("The aggregate stock of assets:")
            print("       A({0}) = {1}".format(self.R, self.aggregate_asset))

        
# ======== The following functions are used to compare multiple instances ==========
def draw_graph_for_question_1d(benchmark, alt_spec):
    # Graphics
    fig, ax = plt.subplots(2, 1,  figsize=(12, 8))
    x_label = benchmark.age_vec
    ax[0].plot(x_label, benchmark.c_path_mean,
                c = 'green', lw = 1.5, label = '$c$')
    ax[0].plot(x_label, benchmark.a_prime_path_mean,
                c = 'red',lw = 1.5, label = "$a'$")
    ax[0].plot(x_label, benchmark.Y_path_mean,
                c = 'blue',lw = 1.5, label = '$Y$')
    ax[0].plot(x_label, alt_spec.c_path_mean,
                c = 'green', lw = 1.5, ls = 'dashed')
    ax[0].plot(x_label, alt_spec.a_prime_path_mean,
                c = 'red', lw = 1.5, ls = 'dashed')
    ax[0].plot(x_label, alt_spec.Y_path_mean,
                c = 'blue', lw = 1.5, ls = 'dashed')
    ax[0].set_xlabel("age")
    ax[0].legend(frameon=False)
    
    ax[1].plot(x_label, benchmark.lnc_var_path,
                c = 'green', lw = 1.5, label = 'var(ln c)')
    ax[1].plot(x_label, benchmark.lnY_var_path,
                c = 'blue', lw = 1.5, label = 'var(ln Y)')
    ax[1].plot(x_label, alt_spec.lnc_var_path,
                c = 'green', lw = 1.5, ls = 'dashed')
    ax[1].plot(x_label, alt_spec.lnY_var_path,
                c = 'blue', lw = 1.5, ls = 'dashed')
    ax[1].set_xlabel("age")
    ax[1].legend(frameon=False)
    plt.savefig('Q1(d).png', dpi = 150, bbox_inches='tight', pad_inches=0)
    
def solve_question_2d_for_ZBC(R_a, 
                              R_b,
                              tol_R = 1E-3, 
                              tol_A = 0,
                              max_iter = 50
                              ):
    model_a = KV2010(R = R_a)
    model_a.discretize_z_process()
    model_a.value_func_iter()
    model_a.solve_for_distribution()
    model_a.calc_aggregate_asset()

    model_b = KV2010(R = R_b)
    model_b.discretize_z_process()
    model_b.value_func_iter()
    model_b.solve_for_distribution()
    model_b.calc_aggregate_asset()
    
    if (model_a.aggregate_asset > tol_A) & (model_b.aggregate_asset > tol_A):
        raise Exception('Both R imply positive amount of aggregate asset.')
    
    if model_a.aggregate_asset > tol_A:
        # Rename so that A(A_a) = 0 and A(A_b) > 0 
        model_a, model_b = model_b, model_a
        R_a, R_b = R_b, R_a
    
    # initialize while loop
    iteration = 0
    to_be_continued = True
    
    stopwatch = StopWatch()
    while to_be_continued:
        R_c = 0.5 * (R_a + R_b)
        print('Iteration {0}: Solving the model with R = {1}...\n'.format(iteration+1, R_c))
        model_c = KV2010(R = R_c)
        model_c.discretize_z_process()
        model_c.value_func_iter()
        model_c.solve_for_distribution()
        model_c.calc_aggregate_asset()
        
        if model_c.aggregate_asset > tol_A:
            R_b = deepcopy(R_c)
        if model_c.aggregate_asset <= tol_A:
            if abs(R_a - R_b) <= tol_R:
                to_be_continued = False
            else:
                R_a = deepcopy(R_c)
        
        iteration += 1
        if iteration == max_iter:
            to_be_continued = False
    
    stopwatch.stop()
    print('Obtained R = {0}, which yields A(R) = {1}\n'.format(R_c, model_c.aggregate_asset))
    return R_c, model_c.aggregate_asset

def solve_question_2d_for_NBC(R_a, 
                              R_b,
                              a_LB,
                              tol_R = 1E-3,
                              tol_A = 1E-6, 
                              max_iter = 50
                              ):
    model_a = KV2010(R = R_a,  a_lb = a_LB)
    model_a.discretize_z_process()
    model_a.value_func_iter()
    model_a.solve_for_distribution()
    model_a.calc_aggregate_asset()

    model_b = KV2010(R = R_b,  a_lb = a_LB)
    model_b.discretize_z_process()
    model_b.value_func_iter()
    model_b.solve_for_distribution()
    model_b.calc_aggregate_asset()

    if abs(model_a.aggregate_asset) < tol_A:
        return R_a, model_a.aggregate_asset
    
    if abs(model_b.aggregate_asset) < tol_A:
        return R_b, model_b.aggregate_asset

    if model_a.aggregate_asset * model_b.aggregate_asset > 0:
        raise Exception('The amount of aggregate asset for each R has the same sign.')
    
    if model_a.aggregate_asset > tol_A:
        # Rename so that A(A_a) < 0 and A(A_b) > 0 
        model_a, model_b = model_b, model_a
        R_a, R_b = R_b, R_a
    
    # initialize while loop
    iteration = 0
    to_be_continued = True
    
    stopwatch = StopWatch()
    while to_be_continued:
        R_c = 0.5 * (R_a + R_b)
        print('Iteration {0}: Solving the model with R = {1}...\n'.format(iteration+1, R_c))
        model_c = KV2010(R = R_c, a_lb = a_LB)
        model_c.discretize_z_process()
        model_c.value_func_iter()
        model_c.solve_for_distribution()
        model_c.calc_aggregate_asset()
        
        if abs(model_c.aggregate_asset) < tol_A:
            to_be_continued = False
        else:
            if abs(R_a - R_b) < tol_R:
                to_be_continued = False
            elif model_c.aggregate_asset < 0:
                R_a = deepcopy(R_c)
            else:
                R_b = deepcopy(R_c)

        iteration += 1
        if iteration == max_iter:
            to_be_continued = False
    
    stopwatch.stop()
    print('Obtained R = {0}, which yields A(R) = {1}\n'.format(R_c, model_c.aggregate_asset))
    return R_c, model_c.aggregate_asset