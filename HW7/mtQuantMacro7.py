#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW7.py

is the python class for the assignment #6 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Oct 25, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import numpy as np
from numpy.random import seed, uniform
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy

# Load the personal Python package, which is based on the assignments #1-#5.
from mtPyTools import StopWatch, find_nearest_idx
from mtPyEcon import gini_index

class AMS2019:
    def __init__(self,
                 sigma = 2.000, # inve
                 r     = 0.030, # net risk-free interest rate,
                 eta   = 0.250, # roll-over interest rate on delinquent debt
                 beta  = 0.900, # discount factor
                 gamma = 0.150, # discharge probability
                 kappa = 0.050,
                 tau   = 0.900, # earnings threshold in DQ
                 f     = 0.060,
                 alpha_a  =  0.0960,
                 alpha_b  = -0.0022,
                 z_vec = np.array([-0.1418, -0.0945, -0.0473, 0., 0.0473, 0.0945, 0.1418]),
                 # possible z
                 pi_z = np.array([
                        [0.9868, 0.0132, 0.    , 0.    , 0.    , 0.    , 0.    ],
                        [0.0070, 0.9814, 0.0117, 0.    , 0.    , 0.    , 0.    ],
                        [0.    , 0.0080, 0.9817, 0.0103, 0.    , 0.    , 0.    ],
                        [0.    , 0.    , 0.0091, 0.9819, 0.0091, 0.    , 0.    ],
                        [0.    , 0.    , 0.    , 0.0103, 0.9817, 0.0080, 0.    ],
                        [0.    , 0.    , 0.    , 0.    , 0.0117, 0.9814, 0.0070],
                        [0.    , 0.    , 0.    , 0.    , 0.    , 0.0132, 0.9868]
                    ]), # transition probability matrix of z
                 eps_vec = np.array([-0.1000, -0.0500, 0.0000, 0.0500, 0.1000]),
                 # possible epsilon
                 pi_eps = np.array([0.0668, 0.2417, 0.3829, 0.2417, 0.0668]),
                 # probability of epsilon
                 a_min = -0.800, # lower bound of grid for a
                 a_max =  4.000, # upper bound of grid for a
                 N_a   = 1000,  # # of grid points
                 age_range = (25, 82), # range of age
                 retire_age = 65, # the last working age
                 penalty = 1E-3 # penalty value for negative consumption
                 ):
        # Ensure the sum of probabilities is unity
        pi_z = pi_z / np.tile(np.sum(pi_z, axis=1).reshape(-1, 1), (1, len(z_vec)))
        pi_eps = pi_eps / np.sum(pi_eps)
        
        # Make the grid for a
        a_grid = np.linspace(a_min, a_max, N_a)
        zero_a_idx = find_nearest_idx(0, a_grid)
        a_grid[zero_a_idx] = 0 # ensure there is zero in the gird for a
        # Store the variabes as the instance attributes
        self.sigma, self.r, self.eta, self.beta = sigma, r, eta, beta
        self.gamma, self.kappa, self.tau, self.f = gamma, kappa, tau, f
        self.alpha_a, self.alpha_b = alpha_a, alpha_b
        self.z_vec, self.pi_z, = z_vec, pi_z
        self.eps_vec, self.pi_eps = eps_vec, pi_eps
        self.a_grid = a_grid
        self.N_z, self.N_eps, self.N_a = len(z_vec), len(eps_vec), N_a
        self.N_age = age_range[1] - age_range[0] + 1
        self.age_vec = np.arange(age_range[0], age_range[1]+1)
        self.N_working_age = retire_age - age_range[0] + 1
        self.working_age_vec = np.arange(age_range[0], retire_age+1)
        self.retire_age = retire_age
        self.penalty = penalty
    
    def age_idx(self, age):
        age_idx = age - self.age_vec[0]
        return int(age_idx)
    
    def fn(self, age):
        base = 1 + self.alpha_a * self.age_idx(age) \
               + self.alpha_b * self.age_idx(age)**2
        if base <= 0:
            print(age)
        return np.log(base)
    
    def pension(self, z):
        yR = 0.1 + 0.9 * np.exp(z)
        yR = np.max([yR, 1.2])
        return yR
    
    def y(self, age, z, eps):
        # Calculate income flow
        if age <= self.retire_age:
            income = np.exp(self.fn(age) + z + eps)
        else:
            income = self.pension(z)
        return income
    
    def y_delinqency(self, age, z, eps):
        income = np.exp(self.fn(age) + z + eps)
        income = min([self.tau * self.fn(age), income])
        return income
    
    def E_G(self, age, V, B, D):
        if age > self.retire_age:
            expected_G = V
        elif age == self.retire_age:
            exp_B_V = np.exp((B - V)/self.kappa)
            exp_B_V[exp_B_V == np.inf] = 1E10
            expected_G = V + self.kappa * np.log(exp_B_V + 1)
        else:
            exp_D_V = np.exp((D - V)/self.kappa)
            exp_B_V = np.exp((B - V)/self.kappa)
            exp_D_V[exp_D_V == np.inf] = 1E10
            exp_B_V[exp_B_V == np.inf] = 1E10
            expected_G = V + self.kappa * np.log(exp_D_V + exp_B_V + 1)
        # a0_idx = find_nearest_idx(0, self.a_grid)
        # expected_G[:, :, a0_idx: ] = V[:, :, a0_idx: ]
        return expected_G
    
    def prob_each_option(self, age, V, B, D):
        if age > self.retire_age:
            prob_B, prob_V, prob_D = 0., 1., 0.
        elif age == self.retire_age:
            exp_B_V = np.exp((B - V)/self.kappa)
            exp_B_V[exp_B_V == np.inf] == 1E10
            denominator = exp_B_V + 1
            prob_B = exp_B_V / denominator       # prob. of bankruptcy
            prob_V = 1 / denominator # prob. of repay
            prob_D = 0                     # prob. of delinquency
            # a0_idx = find_nearest_idx(0, self.a_grid)
            # prob_B[:, :, a0_idx: ] = 0
            # prob_V[:, :, a0_idx: ] = 1
        else:
            exp_D_V = np.exp((D - V)/self.kappa)
            exp_B_V = np.exp((B - V)/self.kappa)
            exp_D_V[exp_D_V == np.inf] = 1E10
            exp_B_V[exp_B_V == np.inf] = 1E10
            denominator = exp_D_V + exp_B_V + 1
            prob_B = exp_B_V / denominator # prob. of bankruptcy
            prob_V = 1 / denominator # prob. of repay
            prob_D = exp_D_V / denominator       # prob. of delinquency
            # a0_idx = find_nearest_idx(0, self.a_grid)
            # prob_B[:, :, a0_idx: ] = 0
            # prob_V[:, :, a0_idx: ] = 1
            # prob_D[:, :, a0_idx: ] = 0
        return prob_B, prob_V, prob_D
    
    def E_G_conditional_on_z(self, age, z, E_G_prime):
        if age < self.retire_age:
            pi_z = self.pi_z
        else:
            pi_z = np.eye(self.N_z)
        # find the index
        z_idx  = find_nearest_idx(z, self.z_vec)
        # take the expectation with respect to epsilon
        E_G_wrt_eps = np.sum(
            [pi_eps_i * E_G_prime[i, :, :] for i, pi_eps_i in enumerate(self.pi_eps)],
            axis = 0
            )
        # take the expectation with repect to z'
        return pi_z[z_idx, :] @ E_G_wrt_eps
    
    def Bn(self, age, z, eps, E_G_prime):
        zero_asset_idx = find_nearest_idx(0, self.a_grid)
        # calculate utility flow
        c = self.y(age, z, eps) - self.f
        if not (np.isscalar(c)):
            c[c <= 0] = self.penalty
        elif c <= 0:
            c = self.penalty
        u = c**(1 - self.sigma)/(1 - self.sigma)
        # Calculate expected value conditional on z
        # Note that the individual cannot borrow when getting bankrupt
        expected_G = (self.E_G_conditional_on_z(age, z, E_G_prime))[zero_asset_idx]
        return u + self.beta * expected_G
    
    def Dn(self, age, a, z, eps, E_G_prime):
        zero_asset_idx = find_nearest_idx(0, self.a_grid)
        # calculate utility flow
        c = self.y_delinqency(age, z, eps)
        if not (np.isscalar(c)):
            c[c <= 0] = self.penalty
        elif c <= 0:
            c = self.penalty
        u = c**(1 - self.sigma)/(1 - self.sigma)
        # a' is given by (1 + eta)a
        a_prime = (1 + self.eta) * a
        # Calculate expected value conditional on z
        E_G_wrt_eps = self.E_G_conditional_on_z(age, z, E_G_prime)
        EG_not_discharged = interp(self.a_grid, E_G_wrt_eps, a_prime)
        EG_discharged = E_G_wrt_eps[zero_asset_idx]
        expected_G = (1 - self.gamma) * EG_not_discharged + self.gamma * EG_discharged
        return u + self.beta * expected_G
    
    def Vn(self, age, a, z, eps, E_G_prime, q):
        # if the sample reaches the last age of life, the value of the state is
        # simply the instantaneous utility. And, in that case, the sample is
        # no longer allowrd to borrow.
        if age == self.age_vec[-1]:
            V = self.utility(age, a = a, z = z, eps = eps, q = 0, a_prime = 0)
            a_prime_star = 0
        else:
            is_retired = (age > self.retire_age)
            a_prime = (self.a_grid).reshape(1, -1)
            utility = self.utility( age = age, a = a, z = z, eps = eps,
                                    q = q, a_prime = a_prime, is_retired = is_retired)
            possible_V = utility + self.beta * self.E_G_conditional_on_z(age, z, E_G_prime)
            # Take max
            V = np.nanmax(possible_V)
            a_prime_star_idx = np.nanargmax(possible_V)
            a_prime_star = a_prime[0, a_prime_star_idx]
        return V, a_prime_star
    
    def qn(self, age, z, a_prime, q_prime, prob_V, prob_D):
        if age-1 < self.retire_age:
            pi_z = self.pi_z
        else:
            pi_z = np.eye(self.N_z)
        # Pick up the indices
        a_prime_idx = find_nearest_idx(a_prime, self.a_grid)
        z_idx = find_nearest_idx(z, self.z_vec)
        # expected return: repay
        contrib_V = prob_V[:, :, a_prime_idx]
        # expected return: delinquency
        prob_D = prob_D[:, :, a_prime_idx]
        q_prime = np.array([
            interp(self.a_grid, q_prime[i, :], a_prime) for i in range(self.N_z)
            ])
        q_prime = np.tile(q_prime, (self.N_eps, 1))
        contrib_D = prob_D * ( (1 - self.gamma) * (1 - self.eta) * q_prime)
        # Taking expectation with respect to epsion
        expected_returns = self.pi_eps @ (contrib_V + contrib_D)
        # calculate the asset price today
        q = pi_z[z_idx, :] @ expected_returns /(1 + self.r)
        return q
    
    def utility(self, age, a, z, eps, q, a_prime, is_retired = False):
        # income at the given age
        income = self.y(age, z, eps)
        # Calculate consumption from the budget constraint
        c = a + income - q * a_prime
        # If c is negstive, give a penalty
        if (np.isscalar(c)):
            if c <= 0:
                c = self.penalty
        else:
            c[c <= 0] = self.penalty
        # During old ages, borrowing constaraint is imposed
        if is_retired:
            if not (np.isscalar(c)):
                c[a_prime < 0] = self.penalty
            elif a_prime < 0:
                c = self.penalty
        # Calculate the CES utility
        u = c**(1-self.sigma) / (1-self.sigma)
        return u
    
    def value_func_iter(self):
        # Assign memory for arrays for values
        B_n = np.zeros((self.N_eps, self.N_z, self.N_a))
        D_n = np.zeros((self.N_eps, self.N_z, self.N_a))
        V_n = np.zeros((self.N_eps, self.N_z, self.N_a))
        a_prime_n = np.zeros((self.N_eps, self.N_z, self.N_a))
        # Prepare the terminal value of E_G' (The afterlife is supposed worthless.)
        E_G_prime = np.zeros((self.N_eps, self.N_z, self.N_a))
        # Prepare 4D arrays where the matrices of G, a', q in each state will be stored
        EG_4Darray      = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        B_4Darray       = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        V_4Darray       = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        D_4Darray       = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        probB_4Darray   = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        probV_4Darray   = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        probD_4Darray   = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        a_prime_4Darray = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        q_3Darray       = np.zeros((self.N_age, self.N_z, self.N_a))
        # start stop watch
        stopwatch = StopWatch()
        print('Solving backward the discretized model...\n')
        # Solve backward (with respect to age)
        for age in reversed(self.age_vec):
            age_idx = self.age_idx(age)
            # calculate the values and the optimal b' for each epsilon, z and a
            for eps_idx, eps in enumerate(self.eps_vec):
                for z_idx, z in enumerate(self.z_vec):
                    for a_idx, a in enumerate(self.a_grid):
                        if age <= self.retire_age:
                            B_n[eps_idx, z_idx, a_idx] = self.Bn(age, z, eps, E_G_prime)
                        if age < self.retire_age:
                            D_n[eps_idx, z_idx, a_idx] = self.Dn(age, a, z, eps, E_G_prime)
                        V_n[eps_idx, z_idx, a_idx], a_prime_n[eps_idx, z_idx, a_idx] = \
                            self.Vn(age, a, z, eps, E_G_prime, q_3Darray[age_idx, z_idx, :])
            # Store the oprimal a' when choosing to repay
            a_prime_4Darray[age_idx, :, :, :] = a_prime_n
            # Store the value from each choice
            if age <= self.retire_age:
                B_4Darray[age_idx, :, :, :] = B_n
            if age < self.retire_age:
                D_4Darray[age_idx, :, :, :] = D_n
            V_4Darray[age_idx, :, :, :] = V_n
            # Calculate and store the expected maximized value
            EG_4Darray[age_idx, :, :, :] = self.E_G(age, V_n, B_n, D_n)
            # Calculate and store probabilities of choosing each option
            probB_4Darray[age_idx, :, :, :], probV_4Darray[age_idx, :,  :, :], probD_4Darray[age_idx, :, :, :] = \
                self.prob_each_option(age, V_n, B_n, D_n)
            # Calculate and store the bond price in the previous period
            if age > self.age_vec[0]:
                q_before = np.array([
                    [self.qn(age, z_i, a_j, q_3Darray[age_idx, :, :], probV_4Darray[age_idx, :, :, :], probD_4Darray[age_idx ,:, :, :])
                    for a_j in self.a_grid]
                    for z_i in self.z_vec
                    ])
                q_3Darray[age_idx-1, :, :] = q_before
            # Use the calculated V as V' in the next loop
            E_G_prime = deepcopy(EG_4Darray[age_idx, :, :, :])
            if age % 10 == 0:
                print(f'Age {age} done.')
            if age == self.retire_age:
                stopwatch.lap()
        stopwatch.stop()
        #Store the solution
        self.EG_data = EG_4Darray
        self.B_data = B_4Darray
        self.V_data = V_4Darray
        self.D_data = D_4Darray
        self.probB_data = probB_4Darray
        self.probV_data = probV_4Darray
        self.probD_data = probD_4Darray
        self.a_prime_data = a_prime_4Darray
        self.q_data = q_3Darray
    
    def get_stochastic_path(self, init_z_idx, init_eps_idx):
        # Prepare the nested functions
        cumsum_pi_z   = np.cumsum(self.pi_z, axis = 1)
        cumsum_pi_eps = np.cumsum(self.pi_eps)
        # Prepare the index vectors for sample z and epsilon
        # (The vectors contain indices, so dtype is a kind of integer)
        eps_path_idx    = np.zeros(self.age_vec.shape, dtype = np.uint8)
        eps_path_idx[0] = init_eps_idx
        z_path_idx   = np.zeros(self.age_vec.shape, dtype = np.uint8)
        z_path_idx[0]= init_z_idx
        for i in range(1, self.N_age, 1):
            if i < len(self.working_age_vec):
                # If the age (i) is in working age, draw z' and epsilon'
                eps_path_idx[i] = draw_exo_idx(cumsum_pi_eps)
                z_path_idx[i]   = draw_exo_idx(cumsum_pi_z[z_path_idx[i-1]])
            else:
                # If the age is in old age, fix z at the level when retiring
                # Note that while epsilon is no longer irrelevant to old ages,
                # we repeat epsilon as well in order to simplify the script
                eps_path_idx[i] = eps_path_idx[i-1]
                z_path_idx[i] = z_path_idx[i-1]
        return eps_path_idx, z_path_idx
    
    def simulate_single_sample(self,
                                init_a = 0, # initial asset holding
                                init_z_idx = 3, # the index of z at age 25
                                init_eps_idx = 2 # the index of epsilon at age 25
                                ):
        # Draw random epsion and z
        eps_path_idx, z_path_idx = \
            self.get_stochastic_path(init_z_idx = init_z_idx,
                                     init_eps_idx = init_eps_idx)
        random_action_choice = uniform(0, 1, self.N_age)
        # Simulate the subject's life using the drawn shocks
        action_path = np.ones(self.age_vec.shape, dtype = np.uint8)
        a_prime_path = np.zeros(self.age_vec.shape)
        c_path = np.zeros(self.age_vec.shape)
        y_path = np.zeros(self.age_vec.shape)
        # initial value of a
        a_i =  deepcopy(init_a)
        a_idx = find_nearest_idx(a_i, self.a_grid)
        for i, age_i in enumerate(self.age_vec):
            # Pick up the income shocks for this age
            eps_idx, z_idx = eps_path_idx[i], z_path_idx[i]
            eps_i, z_i = self.eps_vec[eps_idx], self.z_vec[z_idx]
            # Determine which action the individual chooses
            choice_pdf = [self.probB_data[i, eps_idx, z_idx, a_idx],
                        self.probV_data[i, eps_idx, z_idx, a_idx],
                        self.probD_data[i, eps_idx, z_idx, a_idx],
                        ]
            choice_cdf = np.cumsum(choice_pdf)
            action_i = int(np.sum(choice_cdf < random_action_choice[i]))
            if action_i == 0:
                y_i = self.y(age_i, z_i, eps_i)
                a_prime_i = 0
                c_i = y_i - self.f
            elif action_i == 1:
                y_i = self.y(age_i, z_i, eps_i)
                a_prime_i = self.a_prime_data[i, eps_idx, z_idx, a_idx]
                q_i = interp(self.a_grid, self.q_data[i, z_idx, :], a_prime_i)
                c_i = y_i + a_i - q_i * a_prime_i
            else:
                y_i = self.y_delinqency(age_i, z_i, eps_i)
                c_i = y_i
                if uniform(0, 1) < self.gamma:
                    a_prime_i = 0
                else:
                    a_prime_i = (1 + self.eta) * a_i
            # Store the simulated values for age i into the output arrays
            action_path[i], a_prime_path[i], c_path[i], y_path[i] =\
                action_i, a_prime_i, c_i, y_i
            # Set a_prime today to a tomorrow
            a_i = deepcopy(a_prime_i)
            a_idx = find_nearest_idx(a_i, self.a_grid)
        return action_path, y_path, a_prime_path, c_path
    
    def monte_carlo_simulation(self,
                               init_a,
                               init_z_idx,
                               init_eps_idx,
                               N_samples = 10_000,
                               ):
        # Prepare matrices for simulation result
        action_path_mat = np.zeros((N_samples, self.N_age))
        y_path_mat = np.zeros((N_samples, self.N_age))
        c_path_mat = np.zeros((N_samples, self.N_age))
        a_prime_path_mat = np.zeros((N_samples, self.N_age))
        # Begin the monte carlo
        print('Running simulation with {0} samples...\n'.format(N_samples))
        stopwatch = StopWatch()
        for i in range(N_samples):
            action_path_i, y_path_i, a_prime_path_i, c_path_i = \
                self.simulate_single_sample(init_a = init_a,
                                            init_z_idx   = init_z_idx,
                                            init_eps_idx = init_eps_idx)
            action_path_mat[i, :] = action_path_i
            y_path_mat[i, :] = y_path_i
            c_path_mat[i, :] = c_path_i
            a_prime_path_mat[i, :] = a_prime_path_i
            if i+1 % 1_000 == 0:
                print('Sample #{0}: Done...'.format(i))
        stopwatch.stop()
        # Store the simulation result as instance attribute
        self.action_path_mat, self.y_path_mat, self.c_path_mat, self.a_prime_path_mat = \
            action_path_mat, y_path_mat, c_path_mat, a_prime_path_mat
    
    def summarize_simulation_result(self):
        # Simplify the notations
        action_mat = self.action_path_mat
        y_mat      = self.y_path_mat
        c_mat      = self.c_path_mat
        a_mat      = self.a_prime_path_mat
        N_samples  = y_mat.shape[0]
        # average c, y, a
        self.y_ave = np.nanmean(y_mat, axis = 0)
        self.c_ave = np.nanmean(c_mat, axis = 0)
        self.a_ave = np.nanmean(a_mat, axis = 0)
        # gini coefficients
        self._calc_gini_coeff()
        # indebt household share
        is_indebt = (a_mat < 0)
        N_indebt = np.nansum(is_indebt, axis = 0)
        self.indebt_ratio = N_indebt / N_samples
        # mean debt/income ratio
        ay_ratio = - a_mat / y_mat
        ay_ratio_sum = np.nansum(is_indebt * ay_ratio, axis = 0)
        aux_N_indebt = deepcopy(N_indebt)
        aux_N_indebt[aux_N_indebt==0] = 1
        self.ay_ratio = ay_ratio_sum / aux_N_indebt
        # share of informal default HH
        is_choosing_D = (action_mat == 2)
        N_D = np.nansum(is_choosing_D, axis = 0)
        self.D_ratio = N_D / N_samples
        # share of  HH
        is_choosing_B = (action_mat == 0)
        N_B = np.nansum(is_choosing_B, axis = 0)
        self.B_ratio = N_B / N_samples
    
    def _calc_gini_coeff(self):
        # consumption
        self.gini_coeff_c = self._gini_coeff_routine(self.c_path_mat)
        # income
        self.gini_coeff_y = self._gini_coeff_routine(self.y_path_mat)
        # asset
        # self.gini_coeff_a = self._gini_coeff_routine(self.a_prime_path_mat)
    
    def _gini_coeff_routine(self, simulated_result):
        N_sample, N_periods = simulated_result.shape
        gini_vec = np.zeros((N_periods,))
        for i in range(N_periods):
            simulated_result_i = simulated_result[:, i]
            simulated_result_i = simulated_result_i[~np.isnan(simulated_result_i)]
            simulated_result_i.sort()
            weight_i = np.ones((len(simulated_result_i), )) / len(simulated_result_i)
            gini_vec[i] = gini_index(simulated_result_i, weight_i)
        return gini_vec
    
    def summarize_pooled_simulation_result(self):
        # Prepare pooled data
        retire_idx = self.age_idx(self.retire_age)
        pooled_y = self.y_path_mat[:, :retire_idx+1].flatten()
        pooled_c = self.c_path_mat[:, :retire_idx+1].flatten()
        pooled_a = self.a_prime_path_mat[:, :retire_idx+1].flatten()
        pooled_action = self.action_path_mat[:, :retire_idx+1].flatten()
        N_pooled = len(pooled_y) # # of pooled samples
        # average c, y, a
        self.y_ave_pooled = np.mean(pooled_y)
        self.c_ave_pooled = np.mean(pooled_c)
        self.a_ave_pooled = np.mean(pooled_a)
        # gini coefficients
        weight = np.ones((N_pooled,)) / N_pooled
        self.gini_coeff_y_pooled = gini_index(pooled_y, weight)
        self.gini_coeff_c_pooled = gini_index(pooled_c, weight)
        # indebt household share
        is_indebt = (pooled_a < 0)
        N_indebt = np.sum(is_indebt)
        self.indebt_ratio_pooled = N_indebt/N_pooled
        # mean debt/income ratio
        ay_ratio = - pooled_a / pooled_y
        ay_ratio_sum = np.sum(is_indebt * ay_ratio)
        self.ay_ratio_pooled = ay_ratio_sum / N_indebt
        # share of informal default HH
        is_choosing_D = (pooled_action == 2)
        N_D = np.sum(is_choosing_D)
        self.D_ratio_pooled = N_D / N_pooled
        # share of  HH
        is_choosing_B = (pooled_action == 0)
        N_B = np.sum(is_choosing_B)
        self.B_ratio_pooled = N_B / N_pooled
    
    def solve_for_distribution(self):
        # Prepare a 4D array for the initial distribution
        # -- The distribution is given a 4D array whose size is
        #        (# of age) *  (# of possible epsilon) * (# of possible z) * (# of possible a)
        population = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        # The distribution at age 25 is assumed as follows.
        # -- Everybody enters without asset/debt and with z4 and epsilon3
        no_asset_idx = find_nearest_idx(0, self.a_grid)
        population[0, 2, 3, no_asset_idx] = 1.
        print('Solving for the distribution...\n')
        stopwatch = StopWatch()
        for age_idx in range(self.N_age - 1):
            population[age_idx+1, :, :, :] = \
                self._solve_for_dist_routine(age_idx, population[age_idx, :, :, :])
        # normalize to ensure the total population is unity
        population = population/np.sum(population)
        stopwatch.stop()
        self.distribution = population
    
    def _solve_for_dist_routine(self, age_idx, pdf_pre):
        # Update the pdf
        pdf_at_the_age = [[
            [self._updated_pdf_element(eps_idx, z_idx, a_idx, age_idx, pdf_pre) for a_idx in range(self.N_a)]
            for z_idx in range(self.N_z)]
            for eps_idx in range(self.N_eps)]
        return np.array(pdf_at_the_age)
    
    def _updated_pdf_element(self, eps_idx, z_idx, a_idx, age_idx, pdf_pre):
        # Simplify notation
        a       = self.a_grid
        a_prime = self.a_prime_data[age_idx, :, :, :] # policy function
        a_prime_dq = np.tile(a, (self.N_z, 1)) * (1 + self.eta)
        prob_B  = self.probB_data[age_idx, :, :, :]
        prob_V  = self.probV_data[age_idx, :, :, :]
        prob_D  = self.probD_data[age_idx, :, :, :]
        pi_z    = self.pi_z.T # Pick up the transition matrix for z
        pi_eps  = self.pi_eps # pick up the transition probabilities for epsilon
        def transitL_via_repay_from_eps_i(i):
            if a_idx == 0:
                contrib_i = 0
            else:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        self.indicator(a_prime[i, :, :], a_idx)
                        * (a_prime[i, :, :] - a[a_idx-1])/(a[a_idx] - a[a_idx-1])
                        * prob_V[i, :, :] * pdf_pre[i, : , :]
                        ))
            return contrib_i
        def transitH_via_repay_from_eps_i(i):
            if a_idx == self.N_a - 1:
                contrib_i = 0
            else:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        self.indicator(a_prime[i, :, :], a_idx)
                        * (a[a_idx-1] - a_prime[i, :, :])/(a[a_idx-1] - a[a_idx])
                        * prob_V[i, :, :] * pdf_pre[i, : , :]
                        ))
            return contrib_i
        def transit_via_bankrupt_from_eps_i(i):
            if a[a_idx] == 0:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        prob_B[i, :, :] * pdf_pre[i, : , :]
                        ))
            else:
                contrib_i = 0
            return contrib_i
        def transit_via_discharge_from_eps_i(i):
            if a[a_idx] == 0:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        self.gamma * prob_D[i, :, :] * pdf_pre[i, : , :]
                        ))
            else:
                contrib_i = 0
            return contrib_i
        def transitL_via_overdue_from_eps_i(i):
            if a_idx == 0:
                contrib_i = 0
            else:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        self.indicator(a_prime_dq, a_idx)
                        * (a_prime_dq - a[a_idx-1])/(a[a_idx] - a[a_idx-1])
                        * (1 - self.gamma) * prob_D[i, :, :] * pdf_pre[i, : , :]
                        ))
            return contrib_i
        def transitH_via_overdue_from_eps_i(i):
            if a_idx == self.N_a - 1:
                contrib_i = 0
            else:
                contrib_i = np.sum(
                        pi_eps[eps_idx] * pi_z[z_idx, :] @ (
                        self.indicator(a_prime_dq, a_idx)
                        * (a_prime_dq - a[a_idx-1])/(a[a_idx] - a[a_idx-1])
                        * (1 - self.gamma) * prob_D[i, :, :] * pdf_pre[i, : , :]
                        ))
            return contrib_i
        def transit_from_eps_i(i):
            value = 0
            value += transitH_via_repay_from_eps_i(i)
            value += transitL_via_repay_from_eps_i(i)
            value += transit_via_bankrupt_from_eps_i(i)
            value += transit_via_discharge_from_eps_i(i)
            value += transitH_via_overdue_from_eps_i(i)
            value += transitL_via_overdue_from_eps_i(i)
            return value
        # the volume of transition from each epsilon_i
        contribs = [transit_from_eps_i(i) for i in range(self.N_eps)]
        # Return the sum of the above values
        return np.sum(contribs)
    
    def indicator(self, a_array, idx):
        a = self.a_grid
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
    
    def prepare_policy_functions(self):
        aux_age_list = [age_i if age_i < self.retire_age else self.retire_age
                        for age_i in self.age_vec]
        
        # (1) Bankruptcy
        y_B_data = np.array([[[
            [self.y(age_i, z_j, eps_k) for a_i in self.a_grid]
            for z_j in self.z_vec]
            for eps_k in self.eps_vec]
            for age_i in aux_age_list
        ])
        c_B_data = y_B_data - self.f
        a_prime_B_data = np.zeros((self.N_age, self.N_eps, self.N_z, self.N_a))
        # (2) Repay
        y_V_data = deepcopy(y_B_data)
        q = np.array([[[
            interp(self.a_grid, self.q_data[age_i, z_j,:], self.a_prime_data[age_i, eps_k, z_j, :])
            for z_j in range(self.N_z)]
            for eps_k in range(self.N_eps)]
            for age_i in range(self.N_age)
        ])
        a = np.array([[[
            self.a_grid
            for z_j in range(self.N_z)]
            for eps_k in range(self.N_eps)]
            for age_i in range(self.N_age)
        ])
        c_V_data = y_V_data + a - q * self.a_prime_data
        # (3) Deliquency
        y_D_data = np.array([[[
            [self.y_delinqency(age_i, z_j, eps_k) for a_i in self.a_grid]
            for z_j in self.z_vec]
            for eps_k in self.eps_vec]
            for age_i in aux_age_list
        ])
        c_D_data = deepcopy(y_D_data)
        a_prime_D_data = (1 + self.eta) * a
        # Store the results
        self.y_B_data, self.c_B_data, self.a_prime_B_data = y_B_data, c_B_data, a_prime_B_data
        self.y_V_data, self.c_V_data, self.a_prime_V_data = y_V_data, c_V_data, self.a_prime_data
        self.y_D_data, self.c_D_data, self.a_prime_D_data = y_D_data, c_D_data, a_prime_D_data
    
    def distributional_statistics(self):
        # Simplify the notations
        T = self.age_idx(self.retire_age) + 1

        # (1) average c, y, a
        # income
        y_mat = self.distribution * (self.probB_data * self.y_B_data
                + self.probV_data * self.y_V_data + self.probD_data * self.y_D_data)
        self.y_ave_dist = np.sum(y_mat, axis=(1, 2, 3)) / np.sum(self.distribution, axis=(1, 2, 3))
        self.y_ave_dist_pooled = np.sum(y_mat[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # consumption
        c_mat = self.distribution * (self.probB_data * self.c_B_data
                + self.probV_data * self.c_V_data + self.probD_data * self.c_D_data)
        self.c_ave_dist = np.sum(c_mat, axis=(1, 2, 3)) / np.sum(self.distribution, axis=(1, 2, 3))
        self.c_ave_dist_pooled = np.sum(c_mat[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # asset
        a_mat = self.distribution * (self.probB_data * self.a_prime_B_data
                + self.probV_data * self.a_prime_V_data + self.probD_data * self.a_prime_D_data)
        self.a_ave_dist = np.sum(a_mat, axis=(1, 2, 3)) / np.sum(self.distribution, axis=(1, 2, 3))
        self.a_ave_dist_pooled = np.sum(a_mat[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # (2) gini coefficients
        self._calc_gini_coeff_dist(y_mat, c_mat)
        self.gini_coeff_y_dist_pooled = \
            gini_index(y_mat[:T, :, :, :].flatten(), self.distribution[:T, :, :, :].flatten())
        self.gini_coeff_c_dist_pooled =\
            gini_index(c_mat[:T, :, :, :].flatten(), self.distribution[:T, :, :, :].flatten())
        # (3) indebt household share
        N_indebt = self.distribution * (self.probB_data * (self.a_prime_B_data < 0)
                + self.probV_data * (self.a_prime_V_data < 0) + self.probD_data * (self.a_prime_D_data < 0))
        self.indebt_ratio_dist = np.nansum(N_indebt, axis = (1, 2, 3)) /  np.sum(self.distribution, axis=(1, 2, 3))
        self.indebt_ratio_dist_pooled = np.nansum(N_indebt[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # (4) mean debt/income ratio
        ay_B_ratio = (self.a_prime_B_data < 0) * (-self.a_prime_B_data / self.y_B_data)
        ay_V_ratio = (self.a_prime_V_data < 0) * (-self.a_prime_V_data / self.y_V_data)
        ay_D_ratio = (self.a_prime_D_data < 0) * (-self.a_prime_D_data / self.y_D_data)
        ay_ratio = self.distribution * (self.probB_data * ay_B_ratio
                + self.probV_data * ay_V_ratio + self.probD_data * ay_D_ratio)
        self.ay_ratio_dist = np.sum(ay_ratio, axis = (1, 2, 3)) /  np.sum(self.distribution, axis=(1, 2, 3))
        self.ay_ratio_dist_pooled = np.nansum(ay_ratio[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # (5) share of informal default HH
        N_choosing_D = self.probD_data * self.distribution
        self.D_ratio_dist = np.sum(N_choosing_D, axis = (1, 2, 3)) /  np.sum(self.distribution, axis=(1, 2, 3))
        self.D_ratio_dist_pooled = np.sum(N_choosing_D[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
        # (6) share of bankrupt HH
        N_choosing_B = self.probB_data * self.distribution
        self.B_ratio_dist = np.sum(N_choosing_B, axis = (1, 2, 3)) /  np.sum(self.distribution, axis=(1, 2, 3))
        self.B_ratio_dist_pooled = np.sum(N_choosing_B[:T, :, :, :])/ np.sum(self.distribution[:T, :, :, :])
    
    def _calc_gini_coeff_dist(self, y_mat, c_mat):
        # consumption
        self.gini_coeff_c_dist = self._gini_coeff_dist_routine(c_mat, self.distribution)
        # income
        self.gini_coeff_y_dist = self._gini_coeff_dist_routine(y_mat, self.distribution)
        # asset
        # self.gini_coeff_a = self._gini_coeff_routine(self.a_prime_path_mat)
    
    def _gini_coeff_dist_routine(self, data_mat, distribution):
        N_periods = data_mat.shape[0]
        gini_vec = np.zeros((N_periods,))
        for i in range(N_periods):
            data_i = data_mat[i, :, :, :].flatten()
            weight_i = distribution[i, :, :, :].flatten()
            gini_vec[i] = gini_index(data_i, weight_i)
        return gini_vec
    
    def summarize_pooled_simulation_result(self):
        # Prepare pooled data
        retire_idx = self.age_idx(self.retire_age)
        pooled_y = self.y_path_mat[:, :retire_idx+1].flatten()
        pooled_c = self.c_path_mat[:, :retire_idx+1].flatten()
        pooled_a = self.a_prime_path_mat[:, :retire_idx+1].flatten()
        pooled_action = self.action_path_mat[:, :retire_idx+1].flatten()
        N_pooled = len(pooled_y) # # of pooled samples
        # average c, y, a
        self.y_ave_pooled = np.mean(pooled_y)
        self.c_ave_pooled = np.mean(pooled_c)
        self.a_ave_pooled = np.mean(pooled_a)
        # gini coefficients
        weight = np.ones((N_pooled,)) / N_pooled
        self.gini_coeff_y_pooled = gini_index(pooled_y, weight)
        self.gini_coeff_c_pooled = gini_index(pooled_c, weight)
        # indebt household share
        is_indebt = (pooled_a < 0)
        N_indebt = np.sum(is_indebt)
        self.indebt_ratio_pooled = N_indebt/N_pooled
        # mean debt/income ratio
        ay_ratio = - pooled_a / pooled_y
        ay_ratio_sum = np.sum(is_indebt * ay_ratio)
        self.ay_ratio_pooled = ay_ratio_sum / N_indebt
        # share of informal default HH
        is_choosing_D = (pooled_action == 2)
        N_D = np.sum(is_choosing_D)
        self.D_ratio_pooled = N_D / N_pooled
        # share of  HH
        is_choosing_B = (pooled_action == 0)
        N_B = np.sum(is_choosing_B)
        self.B_ratio_pooled = N_B / N_pooled
    
    def solve_question_a(self,
                        age2plot = 66,
                        z2plot   = (2, 6),
                        eps2plot = 3,
                        fname = 'Q1a.png'):
        # Solve the model backward
        self.value_func_iter()
        # Find the indices to be plotted
        age_idx = self.age_idx(age2plot)
        z_idx   = [z2plot[i] - 1 for i in range(2)]
        eps_idx = eps2plot - 1
        #Graph for Q1(a)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        # --- Plot policy function
        ax[0].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', ls = 'dotted', label = '45 degree line')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[0].set_ylabel("$a'$")
        ax[0].set_xlabel("$a$")
        ax[0].legend(frameon=False)
        # --- Plot value function
        ax[1].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[1].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue',label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[1].set_ylabel("$E[G_{"+f"{age2plot}"+"}]$")
        ax[1].set_xlabel("$a$")
        ax[1].legend(frameon=False)
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_b(self,
                        age2plot = 65,
                        z2plot   = (2, 6),
                        eps2plot = 3,
                        fname = 'Q1b.png'):
        # Find the indices to be plotted
        age_idx = self.age_idx(age2plot)
        z_idx   = [z2plot[i] - 1 for i in range(2)]
        eps_idx = eps2plot - 1
        # Graph for Q1(b)
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        # --- Plot value function
        ax[0, 0].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[0, 0].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[0, 0].set_ylabel("$E[G_{"+f"{age2plot}"+"}]$")
        ax[0, 0].set_xlabel("$a$")
        ax[0, 0].legend(frameon=False)
        # --- Plot value function
        ax[1, 0].plot(self.a_grid, self.B_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[1, 0].plot(self.a_grid, self.B_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[1, 0].set_ylabel("$B_{"+f"{age2plot}"+"}$")
        ax[1, 0].set_xlabel("$a$")
        ax[1, 0].legend(frameon=False)
        # --- Plot value function
        ax[0, 1].plot(self.a_grid, self.V_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[0, 1].plot(self.a_grid, self.V_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[0, 1].set_ylabel("$V_{"+f"{age2plot}"+"}$")
        ax[0, 1].set_xlabel("$a$")
        ax[0, 1].legend(frameon=False)
        ax[1, 1].axis("off")
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_c(self,
                        age2plot = 64,
                        z2plot   = (2, 6),
                        eps2plot = 3,
                        fname = 'Q1c.png'):
        # Find the age index to be plotted
        age_idx = self.age_idx(age2plot)
        # Find a's indices over which a is negative
        a_idx = find_nearest_idx(0, self.a_grid)
        z_idx   = [z2plot[i] - 1 for i in range(2)]
        eps_idx = eps2plot - 1
        # Graph for Q1(c)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # --- Plot bond price
        ax.plot(self.a_grid[:a_idx], self.q_data[age_idx, z_idx[0], :a_idx],
                lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax.plot(self.a_grid[:a_idx], self.q_data[age_idx, z_idx[1], :a_idx],
                lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax.set_ylabel("$q$")
        ax.set_xlabel("$a$")
        ax.legend(frameon=False)
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_d(self,
                        age2plot = (25, 35),
                        z2plot   = (2, 6),
                        eps2plot = 3,
                        fname = 'Q1d.png'):
        # Find the indices to be plotted
        age_idx = [self.age_idx(age2plot[i]) for i in range(2)]
        z_idx   = [z2plot[i] - 1 for i in range(2)]
        eps_idx = eps2plot - 1
        # Graph for Q1(b)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        # --- Plot Policy function for age2plot[0]
        ax[0].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', ls = 'dotted', label = '45 degree line')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx[0], eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx[0], eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[0].set_xlabel("$a$")
        ax[0].set_title("$A_{"+f"{age2plot[0]}"+"}$")
        ax[0].legend(frameon=False)
        # --- Plot Policy function for age2plot[1]
        ax[1].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', ls = 'dotted', label = '45 degree line')
        ax[1].plot(self.a_grid, self.a_prime_data[age_idx[1], eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[1].plot(self.a_grid, self.a_prime_data[age_idx[1], eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[1].set_xlabel("$a$")
        ax[1].set_title("$A_{"+f"{age2plot[1]}"+"}$")
        ax[1].legend(frameon=False)
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    
    def solve_question_e(self,
                        init_a = 0.,
                        init_z_idx = 3,
                        init_eps_idx = 2,
                        N_samples = 10_000,
                        seed2use = None,
                        fnames = ('Q1e1.png', 'Q1e2.png')
                               ):
        if type(seed2use) == type(None):
            seed(seed2use)
        # Run Monte Carlo simulation
        self.monte_carlo_simulation(init_a, init_z_idx, init_eps_idx, N_samples)
        # Get the statistics
        self.summarize_simulation_result()
        # Graphics
        # (1) averages
        fig1, ax1 = plt.subplots(3, 1, figsize=(8, 12))
        ax1[0].plot(self.age_vec, self.c_ave,
                    lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
        ax1[0].plot(self.age_vec, self.y_ave,
                    lw = 1.5, c = 'blue', label = 'income')
        ax1[0].plot(self.age_vec, self.a_ave,
                    lw = 3.0, c = 'red', label = 'savings')
        ax1[0].set_title("average levels")
        ax1[0].legend(frameon=False)
        # (2) Gini coeffs
        ax1[1].plot(self.age_vec, self.gini_coeff_c,
                    lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
        ax1[1].plot(self.age_vec, self.gini_coeff_y,
                    lw = 1.5, c = 'blue', label = 'income')
        #ax1[1].plot(self.age_vec, self.gini_coeff_a,
        #            lw = 3.0, c = 'red', label = 'savings')
        ax1[1].set_title("Gini coefficient")
        ax1[1].legend(frameon=False)
        # (3) Share of indebt household
        ax1[2].plot(self.age_vec, self.indebt_ratio,
                    lw = 3.0, c = 'red')
        ax1[2].set_title("Share of indebt individuals")
        plt.savefig(fnames[0], dpi = 100, bbox_inches='tight', pad_inches=0)
        
        # (1) averages
        fig2, ax2 = plt.subplots(3, 1, figsize=(8, 12))
        ax2[0].plot(self.age_vec, self.ay_ratio,
                    lw = 3.0, c = 'red')
        ax2[0].set_title("Average debt/income ratio")
        # (2) Gini coeffs
        ax2[1].plot(self.age_vec, self.D_ratio,
                    lw = 3.0, c = 'red')
        ax2[1].set_title("Share of those choose D")
        # (3) Share of indebt household
        ax2[2].plot(self.age_vec, self.B_ratio,
                    lw = 3.0, c = 'red')
        ax2[2].set_title("Share of those choose B")
        plt.savefig(fnames[1], dpi = 100, bbox_inches='tight', pad_inches=0)
        
    
    def solve_question_f(self):
        # Calculate the statistics for the pooled samples
        self.summarize_pooled_simulation_result()
        # Print the result
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')   
        print('Statistics for the pooled samples')
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')        
        print(f'Average consumption   :   {self.c_ave_pooled}')
        print(f'Average income        :   {self.y_ave_pooled}')
        print(f'Average asset holdings:   {self.a_ave_pooled}')
        print('')
        print(f'Gini coeficient for consumption:   {self.gini_coeff_c_pooled}')
        print(f'Gini coeficient for income     :   {self.gini_coeff_y_pooled}')
        print('')
        print(f'Share of indebt housheold:   {self.indebt_ratio_pooled}')
        print('')
        print(f'Average debt/income ratio:   {self.ay_ratio_pooled}')       
        print('')
        print(f'Share of those chooging D:   {self.D_ratio_pooled}')
        print(f'Share of those chooging B:   {self.B_ratio_pooled}\n')
        
        
    def solve_question_g(self,
                         fnames = ('Q1g1.png', 'Q1g2.png')
                         ):
        if not hasattr(self, "distribution"):
            self.solve_for_distribution()
        self.prepare_policy_functions()
        self.distributional_statistics()
        # Graphics
        # (1) averages
        fig1, ax1 = plt.subplots(3, 1, figsize=(8, 12))
        ax1[0].plot(self.age_vec, self.c_ave_dist,
                    lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
        ax1[0].plot(self.age_vec, self.y_ave_dist,
                    lw = 1.5, c = 'blue', label = 'income')
        ax1[0].plot(self.age_vec, self.a_ave_dist,
                    lw = 3.0, c = 'red', label = 'savings')
        ax1[0].set_title("average levels")
        ax1[0].legend(frameon=False)
        # (2) Gini coeffs
        ax1[1].plot(self.age_vec, self.gini_coeff_c_dist,
                    lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
        ax1[1].plot(self.age_vec, self.gini_coeff_y_dist,
                    lw = 1.5, c = 'blue', label = 'income')
        #ax1[1].plot(self.age_vec, self.gini_coeff_a,
        #            lw = 3.0, c = 'red', label = 'savings')
        ax1[1].set_title("Gini coefficient")
        ax1[1].legend(frameon=False)
        # (3) Share of indebt household
        ax1[2].plot(self.age_vec, self.indebt_ratio_dist,
                    lw = 3.0, c = 'red')
        ax1[2].set_title("Share of indebt individuals")
        plt.savefig(fnames[0], dpi = 100, bbox_inches='tight', pad_inches=0)
        
        # (1) averages
        fig2, ax2 = plt.subplots(3, 1, figsize=(8, 12))
        ax2[0].plot(self.age_vec, self.ay_ratio_dist,
                    lw = 3.0, c = 'red')
        ax2[0].set_title("Average debt/income ratio")
        # (2) Gini coeffs
        ax2[1].plot(self.age_vec, self.D_ratio_dist,
                    lw = 3.0, c = 'red')
        ax2[1].set_ylim([-0.048, 0.048])
        ax2[1].set_title("Share of those choose D")
        # (3) Share of indebt household
        ax2[2].plot(self.age_vec, self.B_ratio_dist,
                    lw = 3.0, c = 'red')
        ax2[2].set_title("Share of those choose B")
        plt.savefig(fnames[1], dpi = 100, bbox_inches='tight', pad_inches=0)
        # Print the result
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
        print('Statistics for the whole population')
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
        print(f'Average consumption   :   {self.c_ave_dist_pooled}')
        print(f'Average income        :   {self.y_ave_dist_pooled}')
        print(f'Average asset holdings:   {self.a_ave_dist_pooled}')
        print('')
        print(f'Gini coeficient for consumption:   {self.gini_coeff_c_dist_pooled}')
        print(f'Gini coeficient for income     :   {self.gini_coeff_y_dist_pooled}')
        print('')
        print(f'Share of indebt housheold:   {self.indebt_ratio_dist_pooled}')
        print('')
        print(f'Average debt/income ratio:   {self.ay_ratio_dist_pooled}')
        print('')
        print(f'Share of those chooging D:   {self.D_ratio_dist_pooled}')
        print(f'Share of those chooging B:   {self.B_ratio_dist_pooled}\n')
        
        
def interp(x, y, x_hat):
    N = len(x)
    i = np.minimum(np.maximum(np.searchsorted(x, x_hat, side='right'), 1), N-1)
    xl, xr = x[i-1], x[i]
    yl, yr = y[i-1], y[i]
    y_hat = yl + (yr - yl)/(xr - xl) * (x_hat - xl)
    above = x_hat > x[-1]
    below = x_hat < x[0]
    y_hat = np.where(above, y[-1] + (y[-1] - y[-2])/(x[-1] - x[-2]) * (x_hat - x[-1]), y_hat)
    y_hat = np.where(below, y[0], y_hat)
    return y_hat

def draw_exo_idx(cdf):
    # Draw a random number
    rand_val = uniform(0, 1)
    # Decide the drawn variable index depending on the random draw
    drawn_idx = np.sum(cdf < rand_val)
    return int(drawn_idx)

def solve_question_2b(benchmark_model, reformed_model, fnames=('Q2b1.png','Q2b2.png')):
    a0_idx = find_nearest_idx(0, benchmark_model.a_grid)
    V25_old = benchmark_model.V_data[0, :, :, a0_idx]
    V25_new = reformed_model.V_data[0, :, :, a0_idx]
    welfare_gain = (V25_new/V25_old - 1) * 100
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['gray', 'orange', 'red', 'purple', 'blue']
    x_label = benchmark_model.z_vec
    for i in range(benchmark_model.N_eps):
        ax.plot(x_label, welfare_gain[i, :],
                c = colors[i], marker = 'o', label = f'$\\varepsilon = \\varepsilon_{i+1}$')
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\%$")
    ax.set_title("Welfare gain")
    ax.legend(frameon=False)
    plt.savefig(fnames[0], dpi = 100, bbox_inches='tight', pad_inches=0)

    G25_old = benchmark_model.EG_data[0, :, :, a0_idx]
    G25_new = reformed_model.EG_data[0, :, :, a0_idx]
    welfare_gain_G = (G25_new/G25_old - 1) * 100
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(benchmark_model.N_eps):
        ax1.plot(x_label, welfare_gain_G[i, :],
                c = colors[i], marker = 'o', label = f'$\\varepsilon = \\varepsilon_{i+1}$')
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$\%$")
    ax1.set_title("Welfare gain")
    ax1.legend(frameon=False)
    plt.savefig(fnames[1], dpi = 100, bbox_inches='tight', pad_inches=0)

    
def solve_question_2c(benchmark_model, 
                      reformed_model,
                      fnames = ('Q2c1.png', 'Q2c2.png')):
    # Pick up the retire age
    T = benchmark_model.age_idx(benchmark_model.retire_age)
    # Pick up the initial distribution
    population_prior = benchmark_model.distribution[:T, :, :, :]
    # Move forward
    population_post = np.array(
        [reformed_model._solve_for_dist_routine(age_i, population_prior[age_i, :, :])
          for age_i in range(len(benchmark_model.working_age_vec) - 1)
          ])
    
    # Simplify the notation
    probB = reformed_model.probB_data[1:T+1, :, :, :]
    probV = reformed_model.probD_data[1:T+1, :, :, :]
    probD = reformed_model.probV_data[1:T+1, :, :, :]
    y_B = reformed_model.y_B_data[1:T+1, :, :, :]
    y_V = reformed_model.y_V_data[1:T+1, :, :, :]
    y_D = reformed_model.y_D_data[1:T+1, :, :, :]
    c_B = reformed_model.c_B_data[1:T+1, :, :, :]
    c_V = reformed_model.c_V_data[1:T+1, :, :, :]
    c_D = reformed_model.c_D_data[1:T+1, :, :, :]
    a_B = reformed_model.a_prime_B_data[1:T+1, :, :, :]
    a_V = reformed_model.a_prime_V_data[1:T+1, :, :, :]
    a_D = reformed_model.a_prime_D_data[1:T+1, :, :, :]

    # (1) average c, y, a
    # income
    y_mat =probB * y_B + probV * y_V + probD * y_D
    y_ave_dist = np.sum( population_post * y_mat, axis=(1, 2, 3)) / np.sum(population_post, axis=(1, 2, 3))
    y_ave_dist_pooled = np.sum( population_post * y_mat)/ np.sum(population_post)
    # consumption
    c_mat = probB * c_B + probV * c_V + probD * c_D
    c_ave_dist = np.sum(population_post * c_mat, axis=(1, 2, 3)) / np.sum(population_post, axis=(1, 2, 3))
    c_ave_dist_pooled = np.sum( population_post * c_mat)/ np.sum(population_post)
    # asset
    a_mat = probB * a_B + probV * a_V + probD * a_D
    a_ave_dist = np.sum( population_post * a_mat, axis=(1, 2, 3)) / np.sum(population_post, axis=(1, 2, 3))
    a_ave_dist_pooled = np.sum( population_post * a_mat)/ np.sum(population_post)
    # (2) gini coefficients
    gini_coeff_y = reformed_model._gini_coeff_dist_routine(y_mat,  population_post)
    gini_coeff_c = reformed_model._gini_coeff_dist_routine(c_mat,  population_post)
    gini_coeff_y_dist_pooled = \
        gini_index(y_mat.flatten(), population_post.flatten())
    gini_coeff_c_dist_pooled =\
        gini_index(c_mat.flatten(), population_post.flatten())
    # (3) indebt household share
    N_indebt = population_post * (
        probB * (a_B < 0) + probV * (a_V < 0) + probD * (a_D < 0))
    indebt_ratio_dist = np.nansum(N_indebt, axis = (1, 2, 3)) /  np.sum(population_post, axis=(1, 2, 3))
    indebt_ratio_dist_pooled = np.nansum(N_indebt[:T, :, :, :])
    # (4) mean debt/income ratio
    ay_B_ratio = (a_B < 0) * (-a_B / y_B)
    ay_V_ratio = (a_V < 0) * (-a_V / y_V)
    ay_D_ratio = (a_D < 0) * (-a_D / y_D)
    ay_ratio = population_post * (probB * ay_B_ratio
            + probV * ay_V_ratio + probD * ay_D_ratio)
    ay_ratio_dist = np.sum(ay_ratio, axis = (1, 2, 3)) /  np.sum(population_post, axis=(1, 2, 3))
    ay_ratio_dist_pooled = np.nansum(ay_ratio)/ np.sum(population_post)
    # (5) share of informal default HH
    N_choosing_D = probD * population_post
    D_ratio_dist = np.sum(N_choosing_D, axis = (1, 2, 3)) /  np.sum(population_post, axis=(1, 2, 3))
    D_ratio_dist_pooled = np.sum(N_choosing_D)/ np.sum(population_post)
    # (6) share of bankrupt HH
    N_choosing_B = probB * population_post
    B_ratio_dist = np.sum(N_choosing_B, axis = (1, 2, 3)) /  np.sum(population_post, axis=(1, 2, 3))
    B_ratio_dist_pooled = np.sum(N_choosing_B)/ np.sum(population_post)
    
    # Graphics
    age_vec = benchmark_model.age_vec[1:T+1]
    # (1) averages
    fig1, ax1 = plt.subplots(3, 1, figsize=(8, 12))
    ax1[0].plot(age_vec, c_ave_dist,
                lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
    ax1[0].plot(age_vec, y_ave_dist,
                lw = 1.5, c = 'blue', label = 'income')
    ax1[0].plot(age_vec, a_ave_dist,
                lw = 3.0, c = 'red', label = 'savings')
    ax1[0].set_title("average levels")
    ax1[0].legend(frameon=False)
    # (2) Gini coeffs
    ax1[1].plot(age_vec, gini_coeff_c,
                lw = 1.5, c = 'green', ls = 'dashed', label = 'consuption')
    ax1[1].plot(age_vec, gini_coeff_y,
                lw = 1.5, c = 'blue', label = 'income')
    #ax1[1].plot(self.age_vec, self.gini_coeff_a,
    #            lw = 3.0, c = 'red', label = 'savings')
    ax1[1].set_title("Gini coefficient")
    ax1[1].legend(frameon=False)
    # (3) Share of indebt household
    ax1[2].plot(age_vec, indebt_ratio_dist,
                lw = 3.0, c = 'red')
    ax1[2].set_title("Share of indebt individuals")
    plt.savefig(fnames[0], dpi = 100, bbox_inches='tight', pad_inches=0)
    
    # (1) averages
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 12))
    ax2[0].plot(age_vec, ay_ratio_dist,
                lw = 3.0, c = 'red')
    ax2[0].set_title("Average debt/income ratio")
    # (2) Gini coeffs
    ax2[1].plot(age_vec, D_ratio_dist,
                lw = 3.0, c = 'red')
    ax2[1].set_title("Share of those choose D")
    # (3) Share of indebt household
    ax2[2].plot(age_vec, B_ratio_dist,
                lw = 3.0, c = 'red')
    ax2[2].set_title("Share of those choose B")
    plt.savefig(fnames[1], dpi = 100, bbox_inches='tight', pad_inches=0)
    # Print the result
    print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print('Statistics for the whole population')
    print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'Average consumption   :   {c_ave_dist_pooled}')
    print(f'Average income        :   {y_ave_dist_pooled}')
    print(f'Average asset holdings:   {a_ave_dist_pooled}')
    print('')
    print(f'Gini coeficient for consumption:   {gini_coeff_c_dist_pooled}')
    print(f'Gini coeficient for income     :   {gini_coeff_y_dist_pooled}')
    print('')
    print(f'Share of indebt housheold:   {indebt_ratio_dist_pooled}')
    print('')
    print(f'Average debt/income ratio:   {ay_ratio_dist_pooled}')
    print('')
    print(f'Share of those chooging D:   {D_ratio_dist_pooled}')
    print(f'Share of those chooging B:   {B_ratio_dist_pooled}\n')
    
    
    
    