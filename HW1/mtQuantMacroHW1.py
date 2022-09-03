#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW1.py

is the python class foe the assignment #1 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 1st, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
from copy import deepcopy

class Det_NCG_Mdl:
    def __init__(self,
                 sigma = 2.000, # risk aversion
                 beta  = 0.960, # discount factor
                 theta = 0.330, # capital share in production function
                 delta = 0.081, # depreciation rate
                 A     = 0.592, # TFP
                 k_0   = 0.750, # Initial value of k
                 ):
        self.sigma = sigma
        self.beta  = beta
        self.theta = theta
        self.delta = delta
        self.A     = A
        self.k_0   = k_0


    def SteadyState(self, PrintResult = True):
        beta  = self.beta
        theta = self.theta
        delta = self.delta
        A     = self.A
        
        # Calculate the steady-state capital stock
        k_ss = (1/beta + delta - 1)**((theta + 1)/(theta - 1)) *\
            A**(-2/(theta - 1)) * theta**(-(theta + 1)/(theta - 1)) * (1-theta)
        
        # Calculate the steady-state labor
        l_ss = (A * (1-theta))**(1/(1+theta)) * k_ss**(theta/(1+theta)) 
        
        # Calculate the steady-state consumption
        c_ss = A * k_ss**theta * l_ss**(1-theta) - delta*k_ss
        
        # Store the steady-state values as attributes
        self.k_ss = k_ss
        self.l_ss = l_ss
        self.c_ss = c_ss
        
        if PrintResult:
            print(\
                  "\n  k_ss = {:.4f}".format(k_ss),\
                  "\n  l_ss = {:.4f}".format(l_ss),\
                  "\n  c_ss = {:.4f}".format(c_ss),\
                  "\n"
                  )


    def Calc_l(self, k_t):
        theta = self.theta
        A     = self.A
        
        # Calculate l_t following the FOC w.r.t. l_t
        l_t = (A * (1-theta))**(1/(1+theta)) * k_t**(theta/(1+theta)) 
        
        return l_t
    
    
    def DoBisection(self,
                    k_t, # k at the current period
                    k_tp2, # k at the day after tomorrow
                    kmin = 0.1, # lower starting point for bijection
                    kmax = 2, # upper starting point for bijection
                    tol  = 10E-5, # torelance level in bijection
                    MaxIter = 100, # Maximum number of iterations
                    ShowProgress = True,
                    MeasureElapsedTime = True,
                    Silent = False):
        if not Silent:
            print(\
                  "\nStarting bisection to find k_1 and l_1..."
                  "\n" 
                  )

        l_t = self.Calc_l(k_t)
        
        k_a = kmin # lower starting point for bijection
        l_a = self.Calc_l(k_a) # corresponding labor input
        
        k_b = kmax # upper starting point for bijection
        l_b = self.Calc_l(k_b) # corresponding labor input
        
        f_a = self.EvalEulerEq(k_t, l_t, k_a, l_a, k_tp2) # residual in Euler eq when k_a is used
        f_b = self.EvalEulerEq(k_t, l_t, k_b, l_b, k_tp2) # residual in Euler eq when k_b is used
        
        if f_a * f_b > 0: # If the initial function values have the same sign, bisection cannot be applied
            raise ValueError("The starting points do not have the opposite signs. Change k_min and k_max and try again.")
        
        diff_i = min(abs(f_a), abs(f_b))
        i      = 0
        ProgressTable = []
        
        if MeasureElapsedTime:
            tic = time.perf_counter() # stopwatch starts
        
        while diff_i > tol and i < MaxIter:
            k_c = (k_a + k_b)/2
            l_c = self.Calc_l(k_c)
            f_c = self.EvalEulerEq(k_t, l_t, k_c, l_c, k_tp2)
            
            ProgressTable.append([i, k_a, k_b, k_c, f_c])
            
            if f_a * f_c > 0: # If f_a and f_c have the same sign, replace k_a with k_c
                k_a = k_c
                l_a = l_c
                f_a = f_c
            else:             # Otherwise, replace k_b with k_c
                k_b = k_c
                l_b = l_c
                f_b = f_c              
            
            diff_i = abs(f_c)
            i += 1
            
        if ShowProgress and not Silent:
            print(tabulate(ProgressTable, headers=['Iter', 'k_a', 'k_b', 'k_c', 'Diff']))
        
        if diff_i > tol: # If difference is still greater than the tolerance level, raise an exception
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
        if MeasureElapsedTime and not Silent:
            toc = time.perf_counter() # stopwatch stops
            ElapsedTime = toc - tic
            print("\n",
                  "Elapsed time: {:.4f} seconds".format(ElapsedTime))
        
        if not Silent:
            print(\
                  "\n",
                  "\n  k_tp1 = {:.4f}".format(k_c),\
                  "\n  l_tp1 = {:.4f}".format(l_c),\
                  "\n"
                  )

        return k_c, l_c
    
    def DoNewton(self,
                 k_t  , # k in the current period     
                 k_tp2, # k on the day after tomorrow
                 k_init   = 0.82,  # starting point for Newton method
                 tol      = 10E-5, # torelance level in Newton method
                 MaxIter  = 100,   # Maximum number of iterations
                 Stepsize = 0.01,  # Stepsize for numerical differenciation
                 ShowProgress = True,
                 MeasureElapsedTime = True,
                 Silent = False):
        if not Silent:
            print(\
                  "\nStarting (quasi-)Newton method to find k_1 and l_1..."
                  "\n" 
                  )
        l_t = self.Calc_l(k_t)
        
        k_tp1  = k_init
        
        diff_i = 1
        i      = 0
        ProgressTable = []
        
        if MeasureElapsedTime:
            tic = time.perf_counter() # stopwatch starts
            
        while diff_i > tol and i < MaxIter:
            
            l_tp1 = self.Calc_l(k_tp1)
            
            f  = self.EvalEulerEq(k_t, l_t, k_tp1, l_tp1, k_tp2)
            df = self.NumericalDiffEuler(k_t, l_t, k_tp1, l_tp1, k_tp2, Stepsize)

            k_tp1_new = k_tp1 - f/df
            l_tp1_new = self.Calc_l(k_tp1_new)
            f_new     = self.EvalEulerEq(k_t, l_t, k_tp1_new, l_tp1_new, k_tp2)
            
            diff_i = abs(f_new)
            
            k_tp1 = k_tp1_new
            l_tp1 = l_tp1_new
            
            ProgressTable.append([i, k_tp1_new, f_new])
            i += 1
            
        if ShowProgress and not Silent:
            print(tabulate(ProgressTable, headers=['Iter', 'k_t', 'Diff']))
        
        if diff_i > tol: # If difference is still greater than the tolerance level, raise an exception
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
                
        if MeasureElapsedTime and not Silent:
            toc = time.perf_counter() # stopwatch stops
            ElapsedTime = toc - tic
            print("\n",
                  "Elapsed time: {:.4f} seconds".format(ElapsedTime))

        if not Silent:
            print(\
                  "\n",
                  "\n  k_t+1 = {:.4f}".format(k_tp1),\
                  "\n  l_t+1 = {:.4f}".format(l_tp1),\
                  "\n"
                  )
        
        return k_tp1, l_tp1
        
    def EvalEulerEq(self, k_t, l_t, k_tp1, l_tp1, k_tp2):
        beta  = self.beta
        A     = self.A
        theta = self.theta
        delta = self.delta

        LHS = self.MUC(k_t, l_t, k_tp1)
        
        RHS = beta * (A * theta * (k_tp1/l_tp1)**(theta-1) + 1 - delta)\
            * self.MUC(k_tp1, l_tp1, k_tp2)
        
        resid = LHS - RHS
        
        return resid

            
    def MUC(self, k_t, l_t, k_tp1):
        A     = self.A
        theta = self.theta
        delta = self.delta
        sigma = self.sigma
        
        lmbd_t = A * k_t**theta * l_t**(1-theta) + k_t * (1- delta)\
            - k_tp1 - 0.5 * l_t**2
        lmbd_t = lmbd_t**(-sigma)
        
        return lmbd_t
    

    def NumericalDiffEuler(self, k_t, l_t, k_tp1, l_tp1, k_tp2, Stepsize = 0.01):
        k_bar_tp1 = k_tp1 + Stepsize
        l_bar_tp1 = self.Calc_l(k_bar_tp1)
        
        f     = self.EvalEulerEq(k_t, l_t, k_tp1, l_tp1, k_tp2)
        f_bar = self.EvalEulerEq(k_t, l_t, k_bar_tp1, l_bar_tp1, k_tp2)
        
        df = (f_bar - f)/Stepsize 
        
        return df
        
    def DoExtendedPath(self, 
                       k_path_init, 
                       tol = 10E-10, 
                       MaxIter = 500,
                       iter2plot = (0, 1, 2, 3, 4),
                       GraphicName = 'Result_of_Extended_Path' 
                       ):
        k_min = 0.1
        k_max = self.k_ss * 1.1
        
        # Get ready for while loop
        diff_i = 1
        i      = 0
        k_path_old = k_path_init
        k_path2plot = []
        l_path2plot = []
        
        while diff_i > tol and i < MaxIter:
            k_path_new = deepcopy(k_path_old)

            if i in iter2plot:
                k_path2plot.append(k_path_old)
                
                l_path_old = [self.Calc_l(k_path_old[j]) for j in range(len(k_path_init))]
                l_path2plot.append(l_path_old)
            
            for t in range(len(k_path_init)-2):
                k_t   = k_path_new[t]
                k_tp2 = k_path_new[t+2]
                
                k_t, _ = self.DoBisection(k_t = k_t, 
                                          k_tp2 = k_tp2, 
                                          kmin = k_min,
                                          kmax = k_max,
                                          Silent = True)
                k_path_new[t + 1] = deepcopy(k_t)
            
            diff_i = [(k_new - k_old)**2 for (k_new, k_old) in zip(k_path_new, k_path_old)]
            diff_i = sum(diff_i)**0.5
            k_path_old = deepcopy(k_path_new)
            i += 1
        
        # Rename the optimal path to be used
        k_path = k_path_new
        k_path2plot.append(k_path)
        
        # Dynamics of labor input
        l_path = [self.Calc_l(k_path[i]) for i in range(len(k_path))]
        l_path2plot.append(l_path)
        
        # Plot the data
        x = range(len(k_path2plot[0]))
        
        fig, ax = plt.subplots(2, 1, figsize=(6,8))
        
        for i in range(len(k_path2plot)):
            if i < len(k_path2plot) - 1:
                ax[0].plot(x, k_path2plot[i], label='Iteration {:,}'.format(iter2plot[i]))
            else:
                ax[0].plot(x, k_path2plot[i], label='Optimal path')               
        ax[0].set_title('Dynamics of capital stock')
        ax[0].legend(frameon=False)
        
        for i in range(len(l_path2plot)):
            if i < len(l_path2plot) - 1:
                ax[1].plot(x, l_path2plot[i], label='Iteration {:,}'.format(iter2plot[i]))
            else:
                ax[1].plot(x, l_path2plot[i], label='Optimal path')               
        ax[1].set_title('Dynamics of labor input')
        ax[1].legend(frameon=False)
        plt.savefig(GraphicName)
                
        # Store the results as attributes
        self.k_path      = k_path
        self.k_path2plot = k_path2plot
        self.l_path      = l_path
        self.l_path2plot = l_path2plot
                
    def CalcDynamics(self, k_path):
        A     = self.A
        theta = self.theta
        delta = self.delta
        
        # Repeat the last element of k_path to enable to calculate the investment path
        k_path.append(k_path[-1])
        
        # Dynamics of labor input
        l_path = [self.Calc_l(k_path[i]) for i in range(len(k_path) - 1)]
        
        # Dynamics of output
        y_path = [A * k_path[i]**theta * l_path[i]**(1-theta) for i in range(len(k_path) - 1)]
        
        # Dynamics of investment
        x_path = [k_path[i+1] - k_path[i] * (1 - delta) for i in range(len(k_path) - 1)]
        
        # Dynamics of consumption
        c_path = [y_path[i] - x_path[i] for i in range(len(k_path) - 1)]
                
        # Dynamics of interest rate (= net return on capital)
        r_path =  [A * theta * ((k_path[i]/l_path[i])**(1-theta)) - delta for i in range(len(k_path) - 1)]
        
        # Store the result as attributes
        self.l_path = l_path
        self.y_path = y_path
        self.x_path = x_path
        self.c_path = c_path
        self.r_path = r_path
        
        