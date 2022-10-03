#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #4 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW4.py

..........................................................................
Create Sep 27, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================

do_Q1a_c = 1
do_Q1d1  = 1
do_Q1d2  = 1
do_Q1e   = 1
do_Q1f   = 1
do_Q2a   = 1
do_Q2b   = 1
do_Q2c   = 1

# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro4 import IFP, IFP_w_taxation
from mtQuantMacro4 import plot_Q2_graph
import seaborn as sns
sns.set()
from copy import deepcopy
from datetime import timedelta

# =========================================================================
# Question 1 (a) - (c).
# =========================================================================
if do_Q1a_c:
    print("\n",
          "\n **********************************",\
          "\n        Question 1 (a)-(c)         ",\
          "\n **********************************",\
          )

    # Create an instance
    model_1a = IFP()
    # Discretize the income process by Rouwehhorst method
    model_1a.discretize(method='Rouwenhorst')
    # solve the model by policy iteration
    model_1a.policy_func_iter(fname='fig_Q1c.png')
    
    print(timedelta(seconds = model_1a.elasped_time))

# =========================================================================
# Question 1 (d).
# =========================================================================

if do_Q1d1 | do_Q1d2:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (d)          ",\
          "\n **********************************",\
          )

    # Create instances
    model_1d1 = IFP(n_grids_a=50)
    model_1d1.discretize(method='Rouwenhorst')
    model_1d2 = deepcopy(model_1d1)

if do_Q1d1:
    # Solve the model with piecewise interpolation and bisection
    model_1d1.policy_func_iter_w_intrpl(intrpl_method='piecewise',
                                        solving_method='scipy_bisection',
                                        fname='fig_Q1d1.png')
    print(timedelta(seconds = model_1d1.elasped_time))

if do_Q1d2:
    # Solve the model with piecewise interpolation and 'fsolve' solver
    model_1d2.policy_func_iter_w_intrpl(intrpl_method='piecewise',
                                        solving_method='fsolve',
                                        fname='fig_Q1d2.png')
    print(timedelta(seconds = model_1d2.elasped_time))

# =========================================================================
# Question 1 (e).
# =========================================================================

if do_Q1e:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (e)          ",\
          "\n **********************************",\
          )

    # Create an instance
    model_1e = IFP(n_grids_a=20)
    model_1e.discretize(method='Rouwenhorst')
    
    # Solve the model with radial basis function interpolation and fsolve
    model_1e.policy_func_iter_w_intrpl(intrpl_method='radial_basis',
                                       solving_method='fsolve',
                                        fname='fig_Q1e.png')
    print(timedelta(seconds = model_1e.elasped_time))


# =========================================================================
# Question 1 (f).
# =========================================================================
if do_Q1f:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (f)          ",\
          "\n **********************************",\
          )
    
    # Create and instance
    model_1f = IFP(n_grids_a=100)
    model_1f.discretize(method='Rouwenhorst')
    
    # Solve the model with endogenous grid_method
    model_1f.endogenous_grid_method(intrpl_method='piecewise',
                                        fname='fig_Q1f.png')
    print(timedelta(seconds = model_1f.elasped_time))


# =========================================================================
# Question 2 (a).
# =========================================================================
if any([do_Q2a, do_Q2b, do_Q2c]):
    print("\n",
          "\n **********************************",\
          "\n             Question 2            ",\
          "\n **********************************",\
          )
    model_2ab = IFP_w_taxation()
    model_2ab.discretize(method='Rouwenhorst')
    
if do_Q2a:
    # Plot tax
    model_2ab.plot_income_tax()
    

if do_Q2b:
    # solve the model by policy iteration
    model_2ab.policy_func_iter()
    plot_Q2_graph(model_1a, model_2ab,  0, 'Q2b1.png')
    plot_Q2_graph(model_1a, model_2ab, 10, 'Q2b2.png')
    
if do_Q2c:
    model_2c_woT = IFP(a_lb = -0.2)
    model_2c_woT.discretize(method='Rouwenhorst')
    model_2c_woT.policy_func_iter(fname='Q2_model_wo_T.png')
    
    model_2c_wT  = IFP_w_taxation(a_lb = -0.2)
    model_2c_wT.discretize(method='Rouwenhorst')
    model_2c_wT.policy_func_iter(fname='Q2_model_w_T.png')
    
    plot_Q2_graph(model_2c_woT, model_2c_wT,  0, 'Q2c1.png')
    plot_Q2_graph(model_2c_woT, model_2c_wT, 10, 'Q2c2.png')
    