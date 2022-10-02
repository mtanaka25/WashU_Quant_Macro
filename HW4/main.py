#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #4 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW3.py

..........................................................................
Create Sep 27, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================

do_Q1a_c = False
do_Q1d1  = False
do_Q1d2  = False
do_Q1e   = False
do_Q1f   = True


# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro4 import IFP
import seaborn as sns
sns.set()
from copy import deepcopy

# =========================================================================
# Question (a) - (c).
# =========================================================================
if do_Q1a_c:
    print("\n",
          "\n **********************************",\
          "\n        Question 1 (a)-(c)         ",\
          "\n **********************************",\
          )

    # Create an instance
    model_a = IFP()
    # Discretize the income process by Rouwehhorst method
    model_a.discretize(method='Rouwenhorst')
    # solve the model by policy iteration
    model_a.policy_func_iter()

# =========================================================================
# Question (d).
# =========================================================================

if do_Q1d1 | do_Q1d2:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (d)          ",\
          "\n **********************************",\
          )

    # Create instances
    model_d1 = IFP(n_grids_a=50)
    model_d1.discretize(method='Rouwenhorst')
    model_d2 = deepcopy(model_d1)

if do_Q1d1:
    # Solve the model with piecewise interpolation and bisection
    model_d1.policy_func_iter_w_intrpl(intrpl_method='piecewise',
                                       solving_method='scipy_bisection')

if do_Q1d2:
    # Solve the model with piecewise interpolation and 'fsolve' solver
    model_d2.policy_func_iter_w_intrpl(intrpl_method='piecewise',
                                       solving_method='fsolve')

# =========================================================================
# Question (e).
# =========================================================================

if do_Q1e:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (e)          ",\
          "\n **********************************",\
          )

    # Create an instance
    model_e = IFP(n_grids_a=20)
    model_e.discretize(method='Rouwenhorst')
    
    # Solve the model with radial basis function interpolation and fsolve
    model_e.policy_func_iter_w_intrpl(intrpl_method='radial_basis',
                                      solving_method='fsolve')


# =========================================================================
# Question (f).
# =========================================================================
if do_Q1f:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (f)          ",\
          "\n **********************************",\
          )
    
    # Create and instance
    model_f = IFP(n_grids_a=100)
    model_f.discretize(method='Rouwenhorst')
    
    # Solve the model with endogenous grid_method
    model_f.endogenous_grid_method(intrpl_method='piecewise')
    
