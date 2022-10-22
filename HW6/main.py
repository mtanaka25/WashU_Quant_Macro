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



# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro6 import *
import numpy as np

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
benchmark.solve_question_1b(fix_seed = 6969)

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (c)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_1c(fix_seed = 6969)

# =========================================================================
# Question 1 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (d)          ",\
      "\n **********************************",\
      )
NBC = -np.exp(np.max([benchmark.z_grid_list[i][3] for i in range(benchmark.N_w_age)]))
alt_borrowing_constraint = NBC

more_borrowable = KV2010(a_lb = alt_borrowing_constraint)
more_borrowable.solve_question_1a(fname = 'Q1(d)a.png')
more_borrowable.solve_question_1b(fname = 'Q1(d)b.png', fix_seed = 6969)
more_borrowable.solve_question_1c(fname = 'Q1(d)c.png', fix_seed = 6969)

draw_graph_for_question_1d(benchmark = benchmark,
                           alt_spec = more_borrowable)


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
print('   benchmark       : {0}'.format(benchmark.insurance_coef))
print('   alternative spec: {0}'.format(more_borrowable.insurance_coef))
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