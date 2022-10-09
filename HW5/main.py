#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #5 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW5.py

..........................................................................
Create Oct 5, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================
# Set 1 or True to solve the question.
# Set 0 or False to skip the question.
do_Q1a = 1
do_Q1b = 1
do_Q1c = 1
do_Q1d = 1


# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro5 import SimplifiedArellano2008
from mtQuantMacro5 import PiecewiseIntrpl, PiecewiseIntrpl_MeshGrid, RBFIntrpl
from mtQuantMacro5 import get_nearest_idx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from copy import deepcopy
from datetime import timedelta

# =========================================================================
# Question 1 (a).
# =========================================================================
if do_Q1a:
    print("\n",
          "\n **********************************",\
          "\n           Question 1 (a)          ",\
          "\n **********************************",\
          )
    # Solve the question with kappa=0.03
    kappa003 = SimplifiedArellano2008(kappa=0.03)
    kappa003.discretize_lnA_process(method='t')
    kappa003.solve_two_period_DP()

    # Solve the question with kappa=0.05
    kappa005 = SimplifiedArellano2008(kappa=0.05)
    kappa005.discretize_lnA_process(method='t')
    kappa005.solve_two_period_DP()
    
    # Plot result
    fig, ax = plt.subplots(2, 2, figsize=(12, 16))
    ax[0, 0].plot(kappa003.b_grid, kappa003.V_T[19, :],
                  c ='red', lw = 0, marker = "o", label='$\kappa = 0.03$')
    ax[0, 0].plot(kappa005.b_grid, kappa005.V_T[19, :],
                  c ='blue', lw = 0, marker = "x", label='$\kappa = 0.05$')
    ax[0, 0].set_xlabel('b')
    ax[0, 0].set_title('$V_T$ with $A = {:.3f}$'.format(kappa003.A_grid[19]))
    ax[0, 0].legend(frameon=False)

    ax[0, 1].plot(kappa003.A_grid, kappa003.V_T[:, 29],
                  c ='red', lw = 0, marker = "o", label='kappa = 0.03')
    ax[0, 1].plot(kappa005.A_grid, kappa005.V_T[:, 29],
                  c ='blue', lw = 0, marker = "x", label='kappa = 0.05')
    ax[0, 1].set_xlabel('A')
    ax[0, 1].set_title('$V_T$ with $b = {:.3f}$'.format(kappa003.b_grid[29]))
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(kappa003.b_grid, kappa003.q_Tm1[19, :],
                  c ='red', lw = 0, marker = "o", label='kappa = 0.03')
    ax[1, 0].plot(kappa005.b_grid, kappa005.q_Tm1[19, :],
                  c ='blue', lw = 0, marker = "x", label='kappa = 0.05')
    ax[1, 0].set_xlabel("$b'$")
    ax[1, 0].set_title('$q_{T-1}$' + ' with $A = {:.3f}$'.format(kappa003.A_grid[19]))
    ax[1, 0].legend(frameon=False)

    ax[1, 1].plot(kappa003.A_grid, kappa003.q_Tm1[:, 29],
                  c ='red', lw = 0, marker = "o", label='kappa = 0.03')
    ax[1, 1].plot(kappa005.A_grid, kappa005.q_Tm1[:, 29],
                  c ='blue', lw = 0, marker = "x", label='kappa = 0.05')
    ax[1, 1].set_xlabel("$A$")
    ax[1, 1].set_title('$q_{T-1}$' + " with $b' = {:.3f}$".format(kappa003.b_grid[29]))
    ax[1, 1].legend(frameon=False)
    
    plt.savefig('Q1(a).png', dpi = 150, bbox_inches='tight', pad_inches=0)
    
# =========================================================================
# Question 1 (b).
# =========================================================================
if do_Q1b:
    print("\n",
          "\n **********************************",\
          "\n          Question 1 (b)           ",\
          "\n **********************************",\
          )
    kappa003.solve_problem_b(eps=3000)

# =========================================================================
# Question 1 (c).
# =========================================================================
if do_Q1c:
    print("\n",
          "\n **********************************",\
          "\n          Question 1 (c)           ",\
          "\n **********************************",\
          )
    kappa003.solve_problem_c()

# =========================================================================
# Question 1 (c).
# =========================================================================
if do_Q1d:
    print("\n",
          "\n **********************************",\
          "\n          Question 1 (d)           ",\
          "\n **********************************",\
          )
    kappa003.solve_problem_d()
    