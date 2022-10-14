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



# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro5 import SimplifiedArellano2008
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# =========================================================================
# Question 1 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (a)          ",\
      "\n **********************************",\
      )
# Solve the question with kappa=0.03
kappa003 = SimplifiedArellano2008(kappa=0.03)
kappa003.discretize_lnA_process(method='t')
kappa003.solve_problem_a()

# Solve the question with kappa=0.05
kappa005 = SimplifiedArellano2008(kappa=0.05)
kappa005.discretize_lnA_process(method='t')
kappa005.solve_problem_a()

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
print("\n",
      "\n **********************************",\
      "\n          Question 1 (b)           ",\
      "\n **********************************",\
      )
kappa003.solve_problem_b(eps=3000)

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n          Question 1 (c)           ",\
      "\n **********************************",\
      )
kappa003.solve_problem_c()

# =========================================================================
# Question 1 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n          Question 1 (d)           ",\
      "\n **********************************",\
      )
kappa003.solve_problem_d()

# =========================================================================
# Question 1 (e) and (f).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n         Question 1 (e)(f)         ",\
      "\n **********************************",\
      )
kappa003.solve_problem_ef()

# =========================================================================
# Question 2.
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n             Question 2            ",\
      "\n **********************************",\
      )
kappa003.solve_problem_2(method=['method1', 'method2'])

print("\n",
      "\n **********************************",\
      "\n      Question 2 w/o shock         ",\
      "\n **********************************",\
      )
kappa003.solve_problem_2_wo_shock(method=['method1'])
    