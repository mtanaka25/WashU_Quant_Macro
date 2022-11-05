#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #8 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW8.py

..........................................................................
Create Nov 1, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================
K2_guess = 0.1

# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro8 import *
# =========================================================================
# Question 1 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (a)          ",\
      "\n **********************************",\
      )

# =========================================================================
# Question 1 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (b)          ",\
      "\n **********************************",\
      )
benchmark = TwoPeriodKrusellSmith()
benchmark.solve_question_b(K2_guess = K2_guess)

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (c)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_c(K2_guess = K2_guess)

# =========================================================================
# Question 1 (d).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (d)          ",\
      "\n **********************************",\
      )
benchmark.solve_question_d(K2_guess = K2_guess)

# =========================================================================
# Question 1 (e).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (e)          ",\
      "\n **********************************",\
      )
model_Qe1 = TwoPeriodKrusellSmith(k1_min = 0.05, k1_max = 0.25)
model_Qe1.solve_question_d(K2_guess = benchmark.K2, fname = 'Qe_ref1.png')

model_Qe2 = TwoPeriodKrusellSmith(k1_min = 0.0, k1_max = 0.3)
model_Qe2.solve_question_d(K2_guess = benchmark.K2, fname = 'Qf_ref2.png')

compare_interest_rates(benchmark, model_Qe1, model_Qe2)

# =========================================================================
# Question 1 (f).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (f)          ",\
      "\n **********************************",\
      )
model_Qf1 = TwoPeriodKrusellSmith(eps2_min = 0.9,
                                  eps2_max = 1.1)
model_Qf1.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qf_ref1.png')

model_Qf2 = TwoPeriodKrusellSmith(eps2_min = 0.9,
                                  eps2_max = 1.1,
                                  k1_min = 0.05,
                                  k1_max = 0.25)
model_Qf2.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qf_ref2.png')

model_Qf3 = TwoPeriodKrusellSmith(eps2_min = 0.9,
                                  eps2_max = 1.1,
                                  k1_min = 0.00,
                                  k1_max = 0.30)
model_Qf3.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qf_ref3.png')

compare_interest_rates(model_Qf1, model_Qf2, model_Qf3, fname='Qf.png')

# =========================================================================
# Question 1 (g).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (g)          ",\
      "\n **********************************",\
      )
model_Qg1 = TwoPeriodKrusellSmith(k2_min = 0.0)
model_Qg1.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref1.png')

model_Qg2 = TwoPeriodKrusellSmith(k2_min = 0.0,
                                  k1_min = 0.05,
                                  k1_max = 0.25)
model_Qg2.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref2.png')

model_Qg3 = TwoPeriodKrusellSmith(k2_min = 0.0,
                                  k1_min = 0.0,
                                  k1_max = 0.3)
model_Qg3.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref3.png')

compare_interest_rates(model_Qg1, model_Qg2, model_Qg3, fname='Qg1.png')

model_Qg4 = TwoPeriodKrusellSmith(k2_min = 0.0,
                                  eps2_min = 0.9,
                                  eps2_max = 1.1)
model_Qg4.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref4.png')

model_Qg5 = TwoPeriodKrusellSmith(k2_min = 0.0,
                                  eps2_min = 0.9,
                                  eps2_max = 1.1,
                                  k1_min = 0.05,
                                  k1_max = 0.25)
model_Qg5.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref5.png')

model_Qg6 = TwoPeriodKrusellSmith(k2_min = 0.0,
                                  eps2_min = 0.9,
                                  eps2_max = 1.1,
                                  k1_min = 0.0,
                                  k1_max = 0.3)
model_Qg6.solve_question_d(K2_guess = benchmark.K2, 
                           fname = 'Qg_ref6.png')

compare_interest_rates(model_Qg4, model_Qg5, model_Qg6, fname='Qg2.png')


# =========================================================================
# For reference
# =========================================================================
reference_figure(benchmark, model_Qe1, model_Qe2, model_Qf1, model_Qf2, model_Qf3)

