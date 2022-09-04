#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_Q2.py

is main code for the assignment #1 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW1.py

...............................................................................
Create Sep 2nd, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

from mtQuantMacroHW1 import Det_NCG_Mdl
import matplotlib.pyplot as plt
import pandas as pd
import wbdata
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# =========================================================================
# 2.
# =========================================================================
# Setting
T      = 100
A_list = [0.592, 0.588, 0.561, 0.581, 0.596, 0.601, 0.605] + [0.606 for t in range(7, T + 1)]

# Generate a NCG model instance
model3 = Det_NCG_Mdl(T = T, A_list = A_list)

# -------------------------------------------------------------------------
# (a) Calculate the steady state
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (a)            ",\
      "\n **********************************",\
      )
model3.SteadyState(t = 0)

k_ss_new, l_ss_new, c_ss_new = model3.SteadyState(t = T, PrintResult = False, ReturnResult = True)

k_path_guess = [model3.k_ss + (k_ss_new - model3.k_ss) * t / (T + 1) for t in range(T + 1)]

model3.DoExtendedPath(k_path_guess, k_min = model3.k_ss * 0.5, k_max = k_ss_new * 1.1, GraphicName='ExtendedPath_variableA.png')


# -------------------------------------------------------------------------
# (b-1) Calculate l_0 corresponding to the given k_0
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (b)            ",\
      "\n **********************************",\
      )



# -------------------------------------------------------------------------
# (c) Implement extended path to derive the dynamics of k and l
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (c)            ",\
      "\n **********************************",\
      )
# Connect to FRED
fred = Fred(api_key = '824eaabba782e478f1473dc862757ab9')

# Obtain the necessary data from FRED
Y_obs = fred.get_series('GDPC1')
L_obs = fred.get_series('PAYEMS')
I_obs = fred.get_series('GPDIC1')
C_obs = fred.get_series('PCECC96')

# Obtain the real interest rate date from the World Bank database
R_obs = wbdata.get_series('FR.INR.RINR', country=['USA'])

# Detrend Y, L, I, and C
Y_obs_cyc, Y_obs_tre = hpfilter(Y_obs, 1600)
L_obs_cyc, Y_obs_tre = hpfilter(L_obs, 1600)
I_obs_cyc, Y_obs_tre = hpfilter(I_obs, 1600)
C_obs_cyc, Y_obs_tre = hpfilter(C_obs, 1600)



print(R_obs['2021'])
print(Y_obs_cyc)
print(type(Y_obs))
print(type(R_obs))
print("\n")
