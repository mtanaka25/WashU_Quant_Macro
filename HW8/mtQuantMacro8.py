#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW8.py

is the python class for the assignment #8 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Nov 1, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy
from tabulate import tabulate
from scipy.optimize import bisect

# Load the personal Python package, which is based on the assignments #1-#7.
from mtPyTools import StopWatch

class TwoPeriodKrusellSmith:
    def __init__(self,
                 sig      = 2.00, # inverse elasticity of intertemporal substitution
                 alpha    = 0.36, # exponent in the production function]
                 z        = 1.00, # productivity
                 beta     = 0.90, # subjective discount factor
                 eps1     = 1.00, # initial labor supply (common among the households)
                 eps2_min = 0.50, # min of labor supply in period 2
                 eps2_max = 1.50, # max of labor supply in period 2
                 k1_min   = 0.10, # min of initial capital stock
                 k1_max   = 0.20, # max of initial capital stock
                 k2_min   = -99., # ad-hoc borrowing constraint
                 N_k      = 100 , # # of grid points for k
                 N_eps    = 100 , # # of grid points for epsilon
                 ):
        K1 = np.mean([k1_min, k1_max])
        k1_vec = np.linspace(k1_min, k1_max, N_k)
        eps2_vec = np.linspace(eps2_min, eps2_max, N_eps)
        # Save the variables into the instance attributes
        self.sig, self.alpha, self.z, self.beta = sig, alpha, z, beta
        self.eps1, self.eps2_min, self.eps2_max = eps1, eps2_min, eps2_max
        self.k1_min, self.k1_max, self.K1 = k1_min, k1_max, K1
        self.k2_min, self.N_k, self.N_esp =  k2_min, N_k, N_eps
        self.k1_vec, self.eps2_vec = k1_vec, eps2_vec
    
    def r(self, K):
        return self.alpha * self.z * K**(self.alpha - 1)
    
    def w(self, K):
        return (1 - self.alpha) * self.z * K**self.alpha
    
    def c(self, k, k_next, eps, K):
        r, w = self.r(K), self.w(K)
        c = r * k + w * eps - k_next
        if not(np.isscalar(c)):
            c[c <= 0] = np.nan
        elif c <= 0:
            c = np.nan
        return c
    
    def muc(self, k, k_next, eps, K):
        return self.c(k, k_next, eps, K)**(-self.sig)
    
    def resid_of_euler_equation(self, k2, k1, K2):
        # Left-hand side of the Euler equation
        lhs = self.muc(k = k1, k_next = k2, eps = self.eps1, K = self.K1)
        # Right-hand side of the Euler equation
        possible_muc = self.muc(k = k2, k_next = 0, eps = self.eps2_vec, K = K2)
        rhs = self.beta * np.nanmean(possible_muc)
        # return the residual
        return lhs - rhs
    
    def solve_EE_under_guessed_K2(self,
                                  K2_guess,
                                  tol = 1E-10,
                                  maxiter = 1_000):
        r1, w1 = self.r(self.K1), self.w(self.K1)
        r2, w2 = self.r(K2_guess), self.w(K2_guess)
        # set min of k2 to the natural borrowing limit
        # (To ensure positive consumption, multiple 0.99)
        k2_min = - w2*self.eps2_vec[0]/r2 * 0.99
        # If ad-hoc constrant is given, overwrite the limit
        k2_min = max([k2_min, self.k2_min])
        # set max of k2 to the total income in period 1
        # (To ensure positive consumption, multiple 0.99)
        k2_max_list = (r1 * self.k1_vec + w1 * self.eps1) * 0.99
        # allocate the memory for the optimal k2
        optimal_k2_vec = np.zeros((self.N_k, ))
        for n, k1n in enumerate(self.k1_vec):
            resid_k2min = self.resid_of_euler_equation(k2_min, k1n, K2_guess)
            if resid_k2min >= 0:
                optimal_k2 = k2_min
            else:
                optimal_k2 = bisect(self.resid_of_euler_equation,
                                    a = k2_min,
                                    b = k2_max_list[n],
                                    args = (k1n, K2_guess),
                                    xtol = tol,
                                    maxiter = maxiter)
            optimal_k2_vec[n] = optimal_k2
        return optimal_k2_vec
    
    def solve_for_K2_and_its_distribution(self,
                                          K2_init,
                                          tol_for_K2 = 1E-10,
                                          tol_for_bisec = 1E-10,
                                          maxiter_for_K2 = 1000,
                                          maxiter_for_bisec = 1000
                                          ):
        # initialize while loop
        iteration, diff = 0, tol_for_K2 + 1
        K2_guess = deepcopy(K2_init)
        progress_table = []
        stopwatch = StopWatch()
        while (iteration < maxiter_for_K2) & (diff > tol_for_K2):
            optimal_k2_vec = self.solve_EE_under_guessed_K2(
                                    K2_guess = K2_guess,
                                    tol = tol_for_bisec,
                                    maxiter = maxiter_for_bisec
                                    )
            # Since k1 is uniformly distributed, aggregate K2 is simply
            # calculated by averaging optimal_k2_vec
            K2_updated = np.mean(optimal_k2_vec)
            # convergence criteria
            diff = abs(K2_updated - K2_guess)
            # Move ahead iteration counter
            iteration += 1
            # Store the current status
            progress_table.append([iteration, K2_guess, K2_updated, diff])
            # Total capital cannot be negative. So, use the absolute value
            # as the next guess
            K2_guess = deepcopy(max([abs(K2_updated), 1E-5]))
        # print the progress table
        print(tabulate(progress_table,
                       headers=['Iter', 'K2 (guess)', 'K2 (updated)', 'diff'],
                       floatfmt = '.4f'))
        stopwatch.stop()
        ##Check if K2 has been converged
        if diff > tol_for_K2:
            print(f'Failed to find K2 within {maxiter_for_K2} iterations')
        r2, w2 = self.r(K2_updated), self.w(K2_updated)
        print(f'With the solution K_2 = {K2_guess:.4f},...')
        print(f'   r(K2, 1) = {r2:.4f}')
        print(f'   w(K2, 1) = {w2:.4f}')
        print(f'   NBC      = {-w2 * self.eps2_vec[0]/r2:.4f}')
        print('')
        # Store the result in the instance attribute
        self.k2_vec, self.K2 = optimal_k2_vec, K2_updated
        
    def solve_question_b(self, K2_guess):
        r1, w1 = self.r(self.K1), self.w(self.K1)
        r2, w2 = self.r(K2_guess), self.w(K2_guess)
        print('Under the given parameter values,...')
        print(f'   r(K1, 1) = {r1:.4f}')
        print(f'   w(K1, 1) = {w1:.4f}')
        print(f'With the guess K_2 = {K2_guess},...')
        print(f'   r(K2, 1) = {r2:.4f}')
        print(f'   w(K2, 1) = {w2:.4f}')
        print('')
    def solve_question_c(self,
                         K2_guess,
                         tol = 1E-5,
                         maxiter = 100,
                         fname = 'Qc.png'):
        k2_vec = self.solve_EE_under_guessed_K2(K2_guess = K2_guess,
                                                tol = tol,
                                                maxiter = maxiter)
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.plot(self.k1_vec, k2_vec,
                c = 'red', label = '$k_{i,2}^{*}$')
        ax.set_xlabel('$k_{i, 1}$')
        ax.set_ylabel('$k_{i, 2}$')
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
        
    def solve_question_d(self,
                         K2_guess,
                         tol_for_K2 = 1E-10,
                         tol_for_bisec = 1E-10,
                         maxiter_for_K2 = 1000,
                         maxiter_for_bisec = 1000,
                         fname = 'Qd.png'):
        self.solve_for_K2_and_its_distribution(
                                          K2_init = K2_guess,
                                          tol_for_K2 = tol_for_K2,
                                          tol_for_bisec = tol_for_bisec,
                                          maxiter_for_K2 = maxiter_for_K2,
                                          maxiter_for_bisec = maxiter_for_bisec,
                                          )
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.plot(self.k1_vec, self.k2_vec,
                c = 'red', label = '$k_{i,2}^{*}$')
        ax.set_xlabel('$k_{i, 1}$')
        ax.set_ylabel('$k_{i, 2}$')
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)

def compare_interest_rates(model_S, model_M, model_L, fname='Qe.png'):
    r_data = [model_S.r(model_S.K2),
              model_M.r(model_M.K2),
              model_L.r(model_L.K2)]
    y_max, y_min = max(r_data), min(r_data)
    diff = y_max - y_min
    y_max, y_min = y_max+0.5*diff, y_min-0.5*diff
    # Graphics
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    x_label = [f'(1/12)(0.{n+1}$)^2$' for n in range(3)]
    ax.bar(x_label, r_data)
    ax.set_xlabel('Var($k_{i, 1}$)')
    ax.set_ylabel('$r(K_2, 1)$')
    ax.set_ylim(y_min, y_max)
    plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)

def reference_figure(volkL_voleH, # var(k1): low , var(eps2): high
                     volkM_voleH, # var(k1): mid , var(eps2): high
                     volkH_voleH, # var(k1): high, var(eps2): high
                     volkL_voleL, # var(k1): low , var(eps2): low
                     volkM_voleL, # var(k1): mid , var(eps2): low
                     volkH_voleL  # var(k1): high, var(eps2): low
                     ):
    x_data   = volkL_voleH.k1_vec
    y_dataLH = volkL_voleH.k2_vec
    y_dataMH = volkM_voleH.k2_vec
    y_dataHH = volkH_voleH.k2_vec
    y_dataLL = volkL_voleL.k2_vec
    y_dataML = volkM_voleL.k2_vec
    y_dataHL = volkH_voleL.k2_vec
    # Graphics
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    ax.plot(x_data, y_dataLH,
            c = 'red',    label = 'var($k_1$):L, var($\\varepsilon_2$):H')
    ax.plot(x_data, y_dataMH,
            c = 'orange', label = 'var($k_1$):M, var($\\varepsilon_2$):H')
    ax.plot(x_data, y_dataHH,
            c = 'yellow', label = 'var($k_1$):H, var($\\varepsilon_2$):H')
    ax.plot(x_data, y_dataLL,
            c = 'green',  label = 'var($k_1$):L, var($\\varepsilon_2$):L')
    ax.plot(x_data, y_dataML,
            c = 'blue',   label = 'var($k_1$):M, var($\\varepsilon_2$):L')
    ax.plot(x_data, y_dataHL,
            c = 'purple', label = 'var($k_1$):H, var($\\varepsilon_2$):L')
    ax.set_xlabel('$k_{i, 1}$')
    ax.set_ylabel('$k_{i, 2}$')
    ax.legend(frameon = False)
    plt.savefig('ref_fig_png', dpi = 100, bbox_inches='tight', pad_inches=0)
    