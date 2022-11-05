#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #7 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW7.py

..........................................................................
Create Oct 25, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================
preferred_seed = 1234
f_for_Q2 = 0.3

# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro7 import *

# =========================================================================
# Question 1 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (a)          ",\
      "\n **********************************",\
      )
benchmark = AMS2019()
benchmark.solve_question_a()

# =========================================================================
# Question 1 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (b)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_b()

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (c)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_c()

# =========================================================================
# Question 1 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (d)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_d()

# =========================================================================
# Question 1 (e).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (e)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_e(seed2use=preferred_seed)

# =========================================================================
# Question 1 (f).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (f)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_f()

# =========================================================================
# Question 1 (g).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (g)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_g()


# =========================================================================
# Question 2 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (a)          ",\
      "\n **********************************",\
      )
more_bankruptcy_cost = AMS2019(f = f_for_Q2)
more_bankruptcy_cost.value_func_iter()
more_bankruptcy_cost.solve_question_g(fnames = ('Q2a1.png', 'Q2a2.png'))

# =========================================================================
# Question 2 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (b)          ",\
      "\n **********************************",\
      )
solve_question_2b(benchmark, more_bankruptcy_cost)

# =========================================================================
# Question 2 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (c)          ",\
      "\n **********************************",\
      )
solve_question_2c(benchmark, more_bankruptcy_cost)
