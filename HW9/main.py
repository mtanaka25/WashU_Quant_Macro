#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #9 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW9.py

..........................................................................
Create Nov 9, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""
# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro9 import *
from numpy import linspace
# =========================================================================
# Settings
# =========================================================================
p_init = 1.0
m_init = 0.25
ce_list = linspace(20, 60, 21)

z_process = Quasi_Tauchen()
trans_mat = z_process(write_out=True)

# =========================================================================
# Question 1 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (a)          ",\
      "\n **********************************",\
      )
Hopenhayn = Hopenhayn1992(trans_mat = trans_mat)
Hopenhayn.solve_question_1a(p = p_init)

# =========================================================================
# Question 1 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (b)          ",\
      "\n **********************************",\
      )
Hopenhayn.solve_question_1b()

# =========================================================================
# Question 1 (c).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (c)          ",\
      "\n **********************************",\
      )
Hopenhayn.solve_question_1c()

# =========================================================================
# Question 1 (d)(e).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1 (d)          ",\
      "\n **********************************",\
      )
Hopenhayn.solve_question_1d_1e(m_init = m_init)
N_star = Hopenhayn.N_star

# =========================================================================
# Question 2 (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n         Question 2 (a)(b)         ",\
      "\n **********************************",\
      )
solve_question_2a(trans_mat = trans_mat,
                  ce_values = ce_list)
# =========================================================================
# Question 2 (b).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 2 (c)          ",\
      "\n **********************************",\
      )
solve_question_2c(trans_mat = trans_mat,
                  ce_values = ce_list,
                  N_star = N_star)
