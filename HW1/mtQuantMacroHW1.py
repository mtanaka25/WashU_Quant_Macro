#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtQuantMacroHW1.py

is the python class foe the assignment #1 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

...............................................................................
Create Sep 1st, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
from copy import deepcopy
import seaborn

class Det_NCG_Mdl:
    def __init__(self,
                 sigma  = 2.000, # risk aversion
                 beta   = 0.960, # discount factor
                 theta  = 0.330, # capital share in production function
                 delta  = 0.081, # depreciation rate
                 A_list = 0.592, # List for TFP development
                 k_0    = 0.750, # Initial value of k
                 T      = 150    # Simulation periods
                 ):
        if type(A_list) == float:
            A_list = [A_list for i in range(T + 1)]
        elif type(A_list) == list:
            if len(A_list) < T + 1:
                raise Exception("The given TFP path is shorter than the given simulation periods.")
        else:
            raise Exception("The given TFP path must be an float or a list of float.")
        
        
        self.sigma  = sigma
        self.beta   = beta
        self.theta  = theta
        self.delta  = delta
        self.A_list = A_list
        self.k_0    = k_0
        self.T      = T


    def SteadyState(self, t = 0, PrintResult = True, ReturnResult = False):
        beta  = self.beta
        theta = self.theta
        delta = self.delta
        A     = self.A_list[t]
        
        # Calculate the steady-state capital stock
        k_ss = (1/beta + delta - 1)**((theta + 1)/(theta - 1)) *\
            A**(-2/(theta - 1)) * theta**(-(theta + 1)/(theta - 1)) * (1-theta)
        
        # Calculate the steady-state labor
        l_ss = (A * (1-theta))**(1/(1+theta)) * k_ss**(theta/(1+theta)) 
        
        # Calculate the steady-state consumption
        c_ss = A * k_ss**theta * l_ss**(1-theta) - delta*k_ss

        # Calculate the steady-state output
        y_ss = A * k_ss**theta * l_ss**(1-theta)
        
        # Calculate the steady-state investment
        x_ss = y_ss - c_ss
        
        # Calculate the steady-state interest rate
        r_ss = A * theta * (k_ss / l_ss)**(theta - 1) - delta
        
        if PrintResult:
            print(\
                  "\n  k_ss = {:.4f}".format(k_ss),\
                  "\n  l_ss = {:.4f}".format(l_ss),\
                  "\n  c_ss = {:.4f}".format(c_ss),\
                  "\n  y_ss = {:.4f}".format(y_ss),\
                  "\n  x_ss = {:.4f}".format(x_ss),\
                  "\n  r_ss = {:.4f}".format(r_ss),\
                  "\n"
                  )
        if ReturnResult:
            return k_ss, l_ss, c_ss, y_ss, x_ss, r_ss
        else:
            # Store the steady-state values as attributes
            self.k_ss = k_ss
            self.l_ss = l_ss
            self.c_ss = c_ss
            self.y_ss = y_ss
            self.x_ss = x_ss
            self.r_ss = r_ss


    def Calc_l(self, k_t, t = 0):
        theta = self.theta
        A     = self.A_list[t]
        
        # Calculate l_t following the FOC w.r.t. l_t
        l_t = (A * (1-theta))**(1/(1+theta)) * k_t**(theta/(1+theta)) 
        
        return l_t
    
    
    def DoBisection(self,
                    k_t, # k at the current period
                    k_tp2, # k at the day after tomorrow
                    t    = 0, # period
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

        l_t = self.Calc_l(k_t, t)
        
        k_a = kmin # lower starting point for bijection
        l_a = self.Calc_l(k_a, t) # corresponding labor input
        
        k_b = kmax # upper starting point for bijection
        l_b = self.Calc_l(k_b, t) # corresponding labor input
        
        f_a = self.EvalEulerEq(k_t, l_t, k_a, l_a, k_tp2, t) # residual in Euler eq when k_a is used
        f_b = self.EvalEulerEq(k_t, l_t, k_b, l_b, k_tp2, t) # residual in Euler eq when k_b is used
        
        if f_a * f_b > 0: # If the initial function values have the same sign, bisection cannot be applied
            raise ValueError("The starting points do not have the opposite signs. Change k_min and k_max and try again.")
        
        diff_i = min(abs(f_a), abs(f_b))
        i      = 0
        ProgressTable = []
        
        if MeasureElapsedTime:
            tic = time.perf_counter() # stopwatch starts
        
        while diff_i > tol and i < MaxIter:
            k_c = (k_a + k_b)/2
            l_c = self.Calc_l(k_c, t)
            f_c = self.EvalEulerEq(k_t, l_t, k_c, l_c, k_tp2, t)
            
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
                 t        = 0,     # period
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
        l_t = self.Calc_l(k_t, t)
        
        k_tp1  = k_init
        
        diff_i = 1
        i      = 0
        ProgressTable = []
        
        if MeasureElapsedTime:
            tic = time.perf_counter() # stopwatch starts
            
        while diff_i > tol and i < MaxIter:
            
            l_tp1 = self.Calc_l(k_tp1, t)
            
            f  = self.EvalEulerEq(k_t, l_t, k_tp1, l_tp1, k_tp2, t)
            df = self.NumericalDiffEuler(k_t, l_t, k_tp1, l_tp1, k_tp2, t, Stepsize)

            k_tp1_new = k_tp1 - f/df
            l_tp1_new = self.Calc_l(k_tp1_new, t)
            f_new     = self.EvalEulerEq(k_t, l_t, k_tp1_new, l_tp1_new, k_tp2, t)
            
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
        
    def EvalEulerEq(self, k_t, l_t, k_tp1, l_tp1, k_tp2, t = 0):
        beta  = self.beta
        A     = self.A_list[t]
        theta = self.theta
        delta = self.delta

        LHS = self.MUC(k_t, l_t, k_tp1, t)
        
        RHS = beta * (A * theta * (k_tp1/l_tp1)**(theta-1) + 1 - delta)\
            * self.MUC(k_tp1, l_tp1, k_tp2, t+1)
        
        resid = LHS - RHS
        
        return resid

            
    def MUC(self, k_t, l_t, k_tp1, t):
        A     = self.A_list[t]
        theta = self.theta
        delta = self.delta
        sigma = self.sigma
        
        lmbd_t = A * k_t**theta * l_t**(1-theta) + k_t * (1- delta)\
            - k_tp1 - 0.5 * l_t**2
        lmbd_t = lmbd_t**(-sigma)
        
        return lmbd_t
    

    def NumericalDiffEuler(self, k_t, l_t, k_tp1, l_tp1, k_tp2, t = 0, Stepsize = 0.01):
        k_bar_tp1 = k_tp1 + Stepsize
        l_bar_tp1 = self.Calc_l(k_bar_tp1, t)
        
        f     = self.EvalEulerEq(k_t, l_t, k_tp1, l_tp1, k_tp2, t)
        f_bar = self.EvalEulerEq(k_t, l_t, k_bar_tp1, l_bar_tp1, k_tp2, t)
        
        df = (f_bar - f)/Stepsize 
        
        return df
        
    def DoExtendedPath(self, 
                       k_path_init,
                       k_min = 0.1,
                       k_max = 5,
                       tol = 10E-10, 
                       MaxIter = 10000,
                       iter2plot = (0, 1, 2, 3, 4),
                       GraphicName = 'Result_of_Extended_Path' 
                       ):
        
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
                
                l_path_old = [self.Calc_l(k_path_old[t], t) for t in range(len(k_path_init))]
                l_path2plot.append(l_path_old)
            
            for t in range(len(k_path_init)-2):
                k_t   = k_path_new[t]
                k_tp2 = k_path_new[t+2]
                
                k_tp1, _ = self.DoBisection(k_t    = k_t, 
                                            k_tp2  = k_tp2,
                                            t      = t,
                                            kmin   = k_min,
                                            kmax   = k_max,
                                            Silent = True)
                k_path_new[t+1] = deepcopy(k_tp1)
            
            diff_i = [(k_new - k_old)**2 for (k_new, k_old) in zip(k_path_new, k_path_old)]
            diff_i = sum(diff_i)**0.5
            k_path_old = deepcopy(k_path_new)
            i += 1
        
        if diff_i > tol: # If difference is still greater than the tolerance level, raise an exception
            raise Exception("Extended path failed to find the solution within MaxIter.")
        
        # Rename the optimal path to be used
        k_path = k_path_new
        k_path2plot.append(k_path)
        
        # Dynamics of labor input
        l_path = [self.Calc_l(k_path[t], t) for t in range(len(k_path))]
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
        plt.savefig(GraphicName, bbox_inches='tight', pad_inches=0)
        plt.show()
        
        # Store the results as attributes
        self.k_path      = k_path
        self.k_path2plot = k_path2plot
        self.l_path      = l_path
        self.l_path2plot = l_path2plot

                
    def CalcDynamics(self, k_path):
        A_list = self.A_list
        theta  = self.theta
        delta  = self.delta
        
        # Repeat the last element of k_path to enable to calculate the investment path
        k_path.append(k_path[-1])
        
        # Dynamics of labor input
        l_path = [self.Calc_l(k_path[t], t) for t in range(len(k_path) - 1)]
        
        # Dynamics of output
        y_path = [A_list[t] * k_path[t]**theta * l_path[t]**(1-theta) for t in range(len(k_path) - 1)]
        
        # Dynamics of investment
        x_path = [k_path[t+1] - k_path[t] * (1 - delta) for t in range(len(k_path) - 1)]
        
        # Dynamics of consumption
        c_path = [y_path[t] - x_path[t] for t in range(len(k_path) - 1)]
                
        # Dynamics of interest rate (= net return on capital)
        r_path =  [A_list[t] * theta * ((k_path[t]/l_path[t])**(theta - 1)) - delta for t in range(len(k_path) - 1)]
        
        # Store the result as attributes
        self.l_path = l_path
        self.y_path = y_path
        self.x_path = x_path
        self.c_path = c_path
        self.r_path = r_path
        
        