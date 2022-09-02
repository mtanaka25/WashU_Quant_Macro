#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:51:54 2022

@author: tanapanda
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
        self.k_0    = k_0


    def SteadyState(self, PrintResult = True):
        beta  = self.beta
        theta = self.theta
        delta = self.delta
        A     = self.A
       
        # Calculate the steady-state investment
        k_ss = A**(-theta) * theta**(-(1+theta)) * (1-theta) * \
            (1/beta + delta - 1)**(1+theta)
        k_ss = k_ss**(1/(theta**2 - theta - 1))
        
        
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
                  "\n",
                  "\n **********************************",\
                  "\n        Question 1. (a)            ",\
                  "\n **********************************",\
                  "\n  k_ss = {:.4f}".format(k_ss),\
                  "\n  l_ss = {:.4f}".format(l_ss),\
                  "\n  c_ss = {:.4f}".format(c_ss),\
                  "\n"
                  )


    def Calc_l(self, k_t, isQ1b = False):
        theta = self.theta
        A     = self.A
        
        # Calculate l_t following the FOC w.r.t. l_t
        l_t = (A * (1-theta))**(1/(1+theta)) * k_t**(theta/(1+theta)) 
        
        if isQ1b:
            print(\
                  "\n",
                  "\n **********************************",\
                  "\n      Question 1. (b)-i            ",\
                  "\n **********************************",\
                  "\n  l_0 = {:.4f}".format(l_t)
                  )
        return l_t
    
    
    def DoBisection(self,
                    k_t, # k at the current period
                    k_tp2, # k at the day after tomorrow
                    kmin = 0.1, # lower starting point for bijection
                    kmax = 2, # upper starting point for bijection
                    tol  = 10E-5, # torelance level in bijection
                    MaxIter = 100, # Maximum number of iterations
                    ShowProgress = True,
                    MeasureElapsedTime = True):
        print(\
              "\n",
              "\n **********************************",\
              "\n      Question 1. (b)-ii           ",\
              "\n **********************************",\
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
            
        if ShowProgress:
            print(tabulate(ProgressTable, headers=['Iter', 'k_a', 'k_b', 'k_c', 'Diff']))
        
        if diff_i > tol: # If difference is still greater than the tolerance level, raise an exception
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
        if MeasureElapsedTime:
            toc = time.perf_counter() # stopwatch stops
            ElapsedTime = toc - tic
            print("\n",
                  "Elapsed time: {:.4f} seconds".format(ElapsedTime))
        
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
                 MeasureElapsedTime = True):
        print(\
              "\n",
              "\n **********************************",\
              "\n      Question 1. (b)-iii           ",\
              "\n **********************************",\
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
          
        if ShowProgress:
            print(tabulate(ProgressTable, headers=['Iter', 'k_t', 'Diff']))
        
        if diff_i > tol: # If difference is still greater than the tolerance level, raise an exception
            raise Exception("Bisection failed to find the solution within MaxIter.")
        
                
        if MeasureElapsedTime:
            toc = time.perf_counter() # stopwatch stops
            ElapsedTime = toc - tic
            print("\n",
                  "Elapsed time: {:.4f} seconds".format(ElapsedTime))

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
        
    def DoExtendedPath(self, K_path_init, tol = 10E-5, MaxIter = 500):
        k_min = 0.1
        k_max = self.k_ss * 1.1
        
        
        diff_i = 1
        i      = 0
        
        K_path_old = K_path_init
        
        while diff_i > tol and i < MaxIter:
            K_path_new = deepcopy(K_path_old)
            
            for t in range(len(K_path_init)-2):
                k_t   = K_path_new[t]
                k_tp2 = K_path_new[t+2]
                
                k_t, _ = self.DoBisection(k_t = k_t, 
                                          k_tp2 = k_tp2, 
                                          kmin = k_min,
                                          kmax = k_max,
                                          ShowProgress = False,
                                          MeasureElapsedTime = False)
                K_path_new[t + 1] = deepcopy(k_t)
            
            diff_i = [(k_new - k_old)**2 for (k_new, k_old) in zip(K_path_new, K_path_old)]
            diff_i = sum(diff_i)**0.5
            K_path_old = deepcopy(K_path_new)
            i += 1
        
        return K_path_new
                
            