#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #6 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW5.py

..........................................................................
Create Oct 19, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================
preferred_seed = 1234


# =========================================================================
# Load packages
# =========================================================================
from turtle import color
from mtQuantMacro6 import *
import numpy as np
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
benchmark = KV2010()
benchmark.solve_question_1a()

# =========================================================================
# Question 1 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (b)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_1b(fix_seed = preferred_seed)

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (c)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_1c(fix_seed = preferred_seed)

# =========================================================================
# Question 1 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (d)          ",\
      "\n **********************************",\
      )
nbc = -np.exp(np.max([benchmark.z_grid_list[i][3] for i in range(benchmark.N_w_age)]))

NBC = KV2010(a_lb = nbc)
NBC.solve_question_1a(fname = 'Q1(d)a.png')
NBC.solve_question_1b(fname = 'Q1(d)b.png', fix_seed = preferred_seed)
NBC.solve_question_1c(fname = 'Q1(d)c.png', fix_seed = preferred_seed)

draw_graph_for_question_1d(benchmark = benchmark,
                           alt_spec = NBC)


# =========================================================================
# Question 1 (e).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (e)          ",\
      "\n **********************************",\
      )
print("-------------------------------------------------")
print("kaplan and Violante's (2010) insurance coeffcient")
print("-------------------------------------------------")
print('   ZBC    : {0}'.format(benchmark.insurance_coef))
print('   NBC    : {0}'.format(NBC.insurance_coef))
print('\n')

# =========================================================================
# Question 2 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (a)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_2a()
NBC.solve_question_2a(fnames=('Q2(a)1_NBC.png', 'Q2(a)2_NBC.png'))

# =========================================================================
# Question 2 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (b)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_2b()
NBC.solve_question_2b()

# =========================================================================
# Question 2 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (c)          ",\
      "\n **********************************",\
      )
print('Starting to solve the model with R = 1.00...\n')
R_100 = KV2010(R = 1.00)
R_100.discretize_z_process()
R_100.value_func_iter()
R_100.solve_for_distribution()
R_100.calc_aggregate_asset()

print('Starting to solve the model with R = 0.99...\n')
R_099 = KV2010(R = 0.99)
R_099.discretize_z_process()
R_099.value_func_iter()
R_099.solve_for_distribution()
R_099.calc_aggregate_asset()

print('Starting to solve the model with R = 0.98...\n')
R_098 = KV2010(R = 0.98)
R_098.discretize_z_process()
R_098.value_func_iter()
R_098.solve_for_distribution()
R_098.calc_aggregate_asset()

x_data = [0.98, 0.99, 1.00, 1.01]
y_data = [R_098.aggregate_asset, R_099.aggregate_asset,
          R_100.aggregate_asset, benchmark.aggregate_asset]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(x_data, y_data, color = 'red', marker = "o")
ax.set_xlabel('$R$')
ax.set_ylabel('$A(R)$')
plt.savefig("Q2(c).png", dpi = 150, bbox_inches='tight', pad_inches=0)


print('Starting to solve the NBC model with R = 1.00...\n')
NBC_R_100 = KV2010(R = 1.00, a_lb = nbc)
NBC_R_100.discretize_z_process()
NBC_R_100.value_func_iter()
NBC_R_100.solve_for_distribution()
NBC_R_100.calc_aggregate_asset()

print('Starting to solve the NBC model with R = 0.99...\n')
NBC_R_099 = KV2010(R = 0.99, a_lb = nbc)
NBC_R_099.discretize_z_process()
NBC_R_099.value_func_iter()
NBC_R_099.solve_for_distribution()
NBC_R_099.calc_aggregate_asset()

print('Starting to solve the NBC model with R = 0.98...\n')
NBC_R_098 = KV2010(R = 0.98, a_lb = nbc)
NBC_R_098.discretize_z_process()
NBC_R_098.value_func_iter()
NBC_R_098.solve_for_distribution()
NBC_R_098.calc_aggregate_asset()

y_data_NBC = [NBC_R_098.aggregate_asset, NBC_R_099.aggregate_asset,
              NBC_R_100.aggregate_asset, NBC.aggregate_asset]
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.plot(x_data, y_data_NBC, color = 'red', marker = "o")
ax1.set_xlabel('$R$')
ax1.set_ylabel('$A(R)$')
plt.savefig("Q2(c)_NBC.png", dpi = 150, bbox_inches='tight', pad_inches=0)

# =========================================================================
# Question 2 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (d)          ",\
      "\n **********************************",\
      )
R_star_ZBC, A_of_R_star_ZBC = solve_question_2d_for_ZBC(R_a=0.3, R_b=0.4)

R_star_NBC, A_of_R_star_NBC = solve_question_2d_for_NBC(R_a=0.95, R_b=0.97, a_LB = nbc)

R_star_NBC_tighter, A_of_R_star_NBC_tighter = solve_question_2d_for_NBC(R_a=0.90, R_b=0.93, a_LB = nbc/2)

R_star_NBC_looser, A_of_R_star_NBC_looser = solve_question_2d_for_NBC(R_a=0.96, R_b=0.99, a_LB = nbc*1.5)