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

class transition_dynamics:
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
                 penalty = 1E-5 # penalty value for negative consumption
                 ):
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
        self.working_age_vec = np.arange(age_range[0], retire_age+1)
        self.retire_age = retire_age
        self.penalty = penalty
    
    def age_idx(self, age):
        age_idx = age - self.age_vec[0]
        return int(age_idx)
    
    def fn(self, age):
        base = 1 + self.alpha_a * self.age_idx(age) \
               + self.alpha_b * self.age_idx(age)**2
        return np.log(base)
    
    def y(self, age, z, eps):
        # nested function to calculate after-retirement income
        def pension(z):
            yR = 0.1 + 0.9 * np.exp(z)
            yR = np.max([yR, 1.2])
            return yR
        # Calculate income flow
        if age <= self.retire_age:
            income = np.exp(self.fn(age) + z + eps)
        else:
            income = pension(z)
        return income
    
    def E_G(self, age, V, B, D):
        if age > self.retire_age:
            expected_G = V
        elif age == self.retire_age:
            exp_V_B = np.exp((V - B)/self.kappa)
            expected_G = B + self.kappa * np.log(exp_V_B + 1)
        else:
            exp_V_D = np.exp((V - D)/self.kappa)
            exp_B_D = np.exp((B - D)/self.kappa)
            expected_G = D + self.kappa * np.log(exp_V_D + exp_B_D + 1)
        return expected_G
    
    def prob_each_option(self, age, V, B, D):
        if age > self.retire_age:
            prob_B, prob_V, prob_D = 0., 1., 0.
        elif age == self.retire_age:
            exp_V_B = np.exp((V - B)/self.kappa)
            denominator = exp_V_B + 1
            prob_B = 1 / denominator       # prob. of bankruptcy
            prob_V = exp_V_B / denominator # prob. of repay
            prob_D = 0                     # prob. of delinquency
        else:
            exp_V_D = np.exp((V - D)/self.kappa)
            exp_B_D = np.exp((B - D)/self.kappa)
            denominator = exp_V_D + exp_B_D + 1
            prob_B = exp_B_D / denominator # prob. of bankruptcy
            prob_V = exp_V_D / denominator # prob. of repay
            prob_D = 1 / denominator       # prob. of delinquency
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
        u = c**(1 - self.sigma)/(1 - self.sigma)
        # Calculate expected value conditional on z
        # Note that the individual cannot borrow when getting bankrupt
        expected_G = (self.E_G_conditional_on_z(age, z, E_G_prime))[zero_asset_idx]
        return u + self.beta * expected_G
    
    def Dn(self, age, a, z, eps, E_G_prime):
        zero_asset_idx = find_nearest_idx(0, self.a_grid)
        # calculate utility flow
        c = max([self.y(age, z, eps), self.tau * self.fn(age)])
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
            # Calculate and store the expected maximized value
            EG_4Darray[age_idx, :, :, :] = self.E_G(age, V_n, B_n, D_n)
            # Calculate and store probabilities of choosing each option
            bbbb, vvvv, dddd =\
                self.prob_each_option(age, V_n, B_n, D_n) 
            probB_4Darray[age_idx, :, :, :], probV_4Darray[age_idx, :,  :, :], probD_4Darray[age_idx, :, :, :] = \
                self.prob_each_option(age, V_n, B_n, D_n)
            # Calculate and store the bond price in the previous period
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
        stopwatch.stop()
        #Store the solution
        self.EG_data = EG_4Darray
        self.probB_data = probB_4Darray
        self.probV_data = probV_4Darray
        self.probD_data = probD_4Darray
        self.a_prime_data = a_prime_4Darray
        self.q_data = q_3Darray
    
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
        ax[1].set_ylabel("$V_{"+f"{age2plot}"+"}$")
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
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        # --- Plot Policy function
        ax[0].plot(self.a_grid, self.a_grid,
                    lw = 0.75, c = 'black', ls = 'dotted', label = '45 degree line')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[0].plot(self.a_grid, self.a_prime_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[0].set_ylabel("$a'$")
        ax[0].set_xlabel("$a$")
        ax[0].legend(frameon=False)
        # --- Plot value function
        ax[1].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[0], :],
                    lw = 1.5, c = 'blue', label='$z = z_{'+f'{z2plot[0]}'+'}$')
        ax[1].plot(self.a_grid, self.EG_data[age_idx, eps_idx, z_idx[1], :],
                    lw = 3, c = 'red', label='$z = z_{'+f'{z2plot[1]}'+'}$')
        ax[1].set_ylabel("$V_{"+f"{age2plot}"+"}$")
        ax[1].set_xlabel("$a$")
        ax[1].legend(frameon=False)
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
                        fname = 'Q1b.png'):
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