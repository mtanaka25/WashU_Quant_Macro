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
    def __init__(self,
                 a_lb      = 0.,  # borrowing constraint
                 a_max     = 300., # max
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
                 penalty   = 1E-5, # penalty consumption if c is negative
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
                                varname = 'z')
        
        # discretize epsilon
        eps_vec = np.array([mu_eps - np.sqrt(var_eps/2), mu_eps + np.sqrt(var_eps/2)])
        
        # Store the model information as instance attributes
        self.beta, self.R, self.sig = beta, R, sig
        self.sig_z0 = np.sqrt(var_z0)
        self.eps_vec = eps_vec
        self.z_process = z_process
        self.age_vec = age_vec
        self.old_age_vec = old_age_vec
        self.working_age_vec = working_age_vec
        self.a_grid = a_grid
        self.penalty = penalty
    
    
    def discretize_z_process(self,
                             method = 'Rouwenhorst',
                             N = 20, # number of grid points
                             Omega = 3, # scale parameter for Tauchen grid
                             is_write_out_result = False,
                             is_quiet = False):
        self.z_process.discretize(method = method,
                                  N = N,
                                  Omega = Omega,
                                  is_write_out_result = is_write_out_result,
                                  is_quiet = is_quiet)
        # For future convienience, take out grid and transition matrix from AR1 instance.
        self.discretization_method = method
        self.z_grid = self.z_process.z_grid
        self.trans_mat = self.z_process.trans_mat
    
    def utility(self, age, a, a_prime, z, eps = 0):
        # income at the given age
        if age >= self.old_age_vec[0]:
            y = self.pension(z) # z: z at age 65
        else:
            y = self.income(age, z, eps) # z: current z
        #print(y)
        # Calculate consumption from the budget constraint
        c = self.R * a + y - a_prime
        if (np.isscalar(c)):
            if c <= 0:
                c = self.penalty
        else:
            c[c<=0] = self.penalty
        
        # Calculate the CES utility
        u = c**(1-self.sig) / (1-self.sig)
        return u
    
    def income(self, age, z, eps):
        lnY = self.income_trend(age) + z + eps
        Y = np.exp(lnY)
        return Y
    
    def pension(self, z):
        pension_flow = 0.7 * np.exp(z + self.income_trend(self.working_age_vec[-1]))
        return pension_flow
    
    def income_trend(self,
                    age,
                    coef0 = -1.0135,
                    coef1 =  0.1086,
                    coef2 = -0.001122):
        income_trend_at_age = np.log(coef0 + coef1*age + coef2 * age**2)
        return income_trend_at_age
    
    def solve_for_V(self, a, z, eps, age, V_prime, interpolate=False):
        # the current z's index in z grid
        z_idx = find_nearest_idx(z, self.z_grid)
        
        # Pick up the transition matrix of z (depending on age)
        if age < self.working_age_vec[-1]:
            if interpolate:
                trans_mat = self.trans_mat_finer
            else:
                trans_mat = self.trans_mat
        else:
            # During the old ages, z is fixed at the level when retiring
            
            trans_mat = np.eye(len(self.z_grid))
        
        # prepare the (horizontal) vector of a'
        if interpolate:
            a_prime = (self.a_finer_grid).reshape(1, -1)
        else:
            a_prime = (self.a_grid).reshape(1, -1)
        
        # Calculate the possible values
        utility = self.utility(a = a, z = z, eps=eps, age=age, a_prime=a_prime)
        expected_V = trans_mat[z_idx, :] @ [0.5 * V_prime[0, :, :] + 0.5 * V_prime[1, :, :]]
        possible_V = utility + self.beta * expected_V
        
        # Take max
        V = np.nanmax(possible_V)
        a_prime_star_idx = np.nanargmax(possible_V)
        a_prime_star = a_prime[0, a_prime_star_idx]
        
        return V, a_prime_star
    
    def value_func_iter(self, interpolate=False, N_z_finer=100, N_a_finer=2000):
        # Prepare the terminal value of V' (The afterlife is supposed worthless.)
        V_prime = np.zeros((len(self.eps_vec), len(self.z_grid), len(self.a_grid)))
        
        # Prepare 4D arrays where the matrices of V and a' for each age and epsilon will be stored
        V_4Darray = np.zeros((len(self.age_vec), len(self.eps_vec), len(self.z_grid), len(self.a_grid)))
        a_prime_4Darray = np.zeros((len(self.age_vec), len(self.eps_vec), len(self.z_grid), len(self.a_grid)))
        
        # When using interpolation, do necessary preparations
        if interpolate:
            # back up the original transition matrix and grid points
            z_grid_original, trans_mat_original = deepcopy(self.z_grid), deepcopy(self.trans_mat)
            
            # Rerun discretization to obtain the finer transition matrix
            # -- Be careful that self.z_gird and self.trans_mat would be overwritten
            self.discretize_z_process(method = self.discretization_method,
                                      N = N_z_finer,
                                      is_write_out_result = False,
                                      is_quiet = True)
            
            # exchange the variable names
            self.z_grid, self.z_finer_grid = z_grid_original, self.z_grid
            
            # Reduce the size of the transition matrix to N_A * N_A_finer
            z_grid_correspondence = find_nearest_idx(self.z_grid, self.z_finer_grid)
            self.trans_mat = self.trans_mat[z_grid_correspondence, :]
            
            # exchange the variable names
            self.trans_mat, self.trans_mat_finer = trans_mat_original, self.trans_mat
            
            # Prepare the finer grid for a
            self.a_finer_grid = np.linspace(self.a_grid[0], self.a_grid[-1], N_a_finer)
        
        # start stop watch
        stopwatch = StopWatch()
        # Solve backward (with respect to age)
        for count, age in enumerate(reversed(self.age_vec)):
            age_idx = - (count + 1)
            if interpolate: # If use interpolation, do so.
                V_prime_0_intrpl = PiecewiseIntrpl_MeshGrid(self.z_grid, self.a_grid, V_prime[0, :, :])
                V_prime_1_intrpl = PiecewiseIntrpl_MeshGrid(self.z_grid, self.a_grid, V_prime[1, :, :])
                if age < self.working_age_vec[-1]:
                    V_prime_0_finer = V_prime_0_intrpl(self.z_finer_grid, self.a_finer_grid)
                    V_prime_1_finer = V_prime_1_intrpl(self.z_finer_grid, self.a_finer_grid)
                else:
                    # After retiring, z does not change. So, interpolate only wrt a.
                    V_prime_0_finer = V_prime_0_intrpl(self.z_grid, self.a_finer_grid)
                    V_prime_1_finer = V_prime_1_intrpl(self.z_grid, self.a_finer_grid)
                V_prime = np.array([V_prime_0_finer, V_prime_1_finer])
            
            # calculate the maximized V and the optimal b' for each epsilon, z and a
            for eps_idx, eps in enumerate(self.eps_vec):
                for z_idx, z in enumerate(self.z_grid):
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
        cumsum_transmat = np.cumsum(self.trans_mat, axis = 1)
        def draw_z_prime_idx(z_idx):
            cumsum_transmat_z = cumsum_transmat[z_idx, :]
            rand_val = uniform(0, 1)
            is_below = (cumsum_transmat_z < rand_val)
            z_prime_idx = np.sum(is_below)
            return int(z_prime_idx)
        def draw_eps_prime_idx():
            return int(round(uniform(0, 1)))
        
        # Prepare the vectors for the sample path
        eps_path_idx    = np.zeros(self.working_age_vec.shape, dtype = np.uint8)
        eps_path_idx[0] = init_eps_idx
        z_path_idx   = np.zeros(self.working_age_vec.shape, dtype = np.uint8)
        z_path_idx[0]= init_z_idx
        
        for i in range(len(self.working_age_vec)-1):
            eps_path_idx[i + 1] = draw_eps_prime_idx()
            z_path_idx[i + 1] = draw_z_prime_idx(z_path_idx[i])
        
        return eps_path_idx, z_path_idx
    
    
    def simulate_single_sample(self,
                                init_a = 0, # initial asset holding
                                init_z_idx = 14, # the index of z at age 25
                                init_eps_idx = 0 # the index of epsilon at age 25
                                ):
        eps_path_idx, z_path_idx = \
            self.get_stochastic_path(init_z_idx = init_z_idx,
                                     init_eps_idx = init_eps_idx)
        # income history
        eps_path = self.eps_vec[eps_path_idx]
        z_path = self.z_grid[z_path_idx]
        Y_path = self.income(age = self.working_age_vec,
                             z = z_path,
                             eps = eps_path)
        # asset and consumption history
        a_prime_path = np.zeros(self.working_age_vec.shape)
        c_path = np.zeros(self.working_age_vec.shape)
        a_idx = find_nearest_idx(init_a, self.a_grid)
        for i in range(len(self.working_age_vec)):
            a_prime_i = self.a_prime[i,
                                    eps_path_idx[i],
                                    z_path_idx[i],
                                    a_idx]
            c_path[i] = self.R * self.a_grid[a_idx] + Y_path[i] - a_prime_i
            a_prime_path[i] = a_prime_i
            a_idx = find_nearest_idx(a_prime_i, self.a_grid)
        return Y_path, a_prime_path, c_path
    
    
    def monte_carlo_simulation(self,
                               init_a,
                               init_z_idx,
                               init_eps_idx,
                               N_samples = 25_000,
                               ):
        # Prepare matrices for simulation result
        mat_size = (N_samples, len(self.working_age_vec))
        Y_path_mat = np.zeros(mat_size)
        c_path_mat = np.zeros(mat_size)
        a_prime_path_mat = np.zeros(mat_size)
        
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
        # convert sample data into growth from the previous age
        dY_mat = (self.Y_path_mat[1:, :] - self.Y_path_mat[:-1, :]).flatten()
        dc_mat = (self.c_path_mat[1:, :] - self.c_path_mat[:-1, :]).flatten()
        # calculate variance and covariace
        var_dY = np.var(dY_mat)
        cov_dY_dc = np.cov(dY_mat, dc_mat)[0, 1]
        # calculate the coefficent
        insurance_coef = 1 - cov_dY_dc/var_dY
        
        self.insurance_coef = insurance_coef
    
    def solve_question_1a(self,
                        method = 'Rouwenhorst',
                        ages2plot = (25, 40, 60),
                        z2plot    = (5, 10, 15),
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
        z_idx    = find_nearest_idx(z2plot, self.z_grid)
        ages_idx = find_nearest_idx(ages2plot, self.age_vec)
        
        fig, ax = plt.subplots(3, 1, figsize=(12, 16))
        
        for i in range(2):
            ax[i].plot(self.a_grid,self.a_grid,
                        lw = 0.75, c = 'black', label = '45 degree line')
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[0], 0, z_idx[i], :].flatten(),
                        lw = 1.5, c = 'gray', ls = 'dashed',
                        label='{0} years old'.format(ages2plot[0]))
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[1], 0, z_idx[i], :].flatten(),
                        lw = 1.5, c = 'blue',
                        label='{0} years old'.format(ages2plot[1]))
            ax[i].plot(self.a_grid,
                        self.a_prime[ages_idx[2], 0, z_idx[i], :].flatten(),
                        lw = 2.5, c = 'red',
                        label='{0} years old'.format(ages2plot[2]))
            ax[i].set_xlabel("$a$")
            ax[i].set_title("$a'(a | z, \\varepsilon_L, age)$")
            ax[i].legend(frameon=False)
        ax[2].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', label = '45 degree line')
        ax[2].plot(self.a_grid, self.a_prime[ages_idx[1], 0, z_idx[2], :].flatten(),
                        lw = 1.5, c = 'blue',
                        label='\\varepsilon_L')
        ax[2].plot(self.a_grid, self.a_prime[ages_idx[1], 1, z_idx[2], :].flatten(),
                        lw = 2.5, c = 'red',
                        label='\\varepsilon_H')
        ax[2].set_xlabel("$a$")
        ax[2].set_title("$a'(a | z_{15}, \\varepsilon, 40)$")
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
            seed = fix_seed
        
        Y_path, a_prime_path, c_path = self.simulate_single_sample(init_a = init_a,
                                                              init_z_idx   = init_z_idx,
                                                              init_eps_idx = init_eps_idx)
        
        fig, ax = plt.subplots(1, 1,  figsize=(12, 8))
        x_label = self.working_age_vec
        ax.plot(x_label, c_path,
                c = 'gray',lw = 2.5, label = 'c')
        ax.plot(x_label, a_prime_path,
                c = 'blue',lw = 1.5, ls = 'dashed', label = "$a'$")
        ax.plot(x_label, Y_path,
                c = 'red',lw = 2., label = 'Y')
        ax.set_xlabel("age")
        ax.legend(frameon=False)
        plt.savefig(fname, dpi = 150, bbox_inches='tight', pad_inches=0)
        
    def solve_question_1c(self,
                            init_a = 0.,
                            init_z_idx = 4,
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
        x_label = self.working_age_vec
        ax[0].plot(x_label, self.c_path_mean,
                   c = 'gray',lw = 2.5, label = 'c')
        ax[0].plot(x_label, self.a_prime_path_mean,
                   c = 'blue',lw = 1.5, ls = 'dashed', label = "$a'$")
        ax[0].plot(x_label, self.Y_path_mean,
                   c = 'red',lw = 2., label = 'Y')
        ax[0].set_xlabel("age")
        ax[0].legend(frameon=False)
        
        ax[1].plot(x_label, self.lnc_var_path,
                   c = 'gray',lw = 2.5, label = 'c')
        ax[1].plot(x_label, self.lnY_var_path,
                   c = 'red',lw = 2., label = 'Y')
        ax[1].set_xlabel("age")
        ax[1].legend(frameon=False)
        plt.savefig(fname, dpi = 150, bbox_inches='tight', pad_inches=0)
        
def draw_graph_for_question_1d(benchmark, alt_spec):
    # Graphics
    fig, ax = plt.subplots(2, 1,  figsize=(12, 8))
    x_label = benchmark.working_age_vec
    ax[0].plot(x_label, benchmark.c_path_mean,
                c = 'green', lw = 1.5, label = 'c: benchmark')
    ax[0].plot(x_label, benchmark.a_prime_path_mean,
                c = 'red',lw = 1.5, label = "$a'$: benchmark")
    ax[0].plot(x_label, benchmark.Y_path_mean,
                c = 'blue',lw = 1.5, label = 'Y: benchmark')
    ax[0].plot(x_label, alt_spec.c_path_mean,
                c = 'green', lw = 1.5, ls = 'dashed',
                label = 'c: alternative spec')
    ax[0].plot(x_label, alt_spec.a_prime_path_mean,
                c = 'red', lw = 1.5, ls = 'dashed',
                label = "$a'$: alternative spec")
    ax[0].plot(x_label, alt_spec.Y_path_mean,
                c = 'blue', lw = 1.5, ls = 'dashed',
                label = 'Y: alternative spec')
    ax[0].set_xlabel("age")
    ax[0].legend(frameon=False)
    
    ax[1].plot(x_label, benchmark.lnc_var_path,
                c = 'green', lw = 1.5, label = 'c: benchmark')
    ax[1].plot(x_label, benchmark.lnY_var_path,
                c = 'blue', lw = 1.5, label = 'Y: benchmark')
    ax[1].plot(x_label, alt_spec.lnc_var_path,
                c = 'green', lw = 1.5, ls = 'dashed',
                label = 'c: alternative spec')
    ax[1].plot(x_label, alt_spec.lnY_var_path,
                c = 'blue', lw = 1.5, ls = 'dashed',
                label = 'Y: alternative spec')
    ax[1].set_xlabel("age")
    ax[1].legend(frameon=False)
    plt.savefig('fig1(d).png', dpi = 150, bbox_inches='tight', pad_inches=0)