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
import numpy as np
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
# (a) Implement extended path to derive the dynamics of k and l
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (a)            ",\
      "\n **********************************",\
      )
# Calculate the initial steady state (under A = 0.592)
model3.SteadyState(t = 0)

# Calcurlte the new steady-state consistent with A = 0.606
k_ss_new, _, _, _, _, _ = model3.SteadyState(t = T, PrintResult = False, ReturnResult = True)

# Prepare a guess for the extended path method. Here, I guess a linear transition.
k_path_guess = [model3.k_ss + (k_ss_new - model3.k_ss) * t / (T + 1) for t in range(T + 1)]

# Implement the extended path
model3.DoExtendedPath(k_path_guess, k_min = model3.k_ss * 0.5, k_max = k_ss_new * 1.1, GraphicName='ExtendedPath_variableA.png')


# -------------------------------------------------------------------------
# (b) Calculate the other variables' dynamics
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (b)            ",\
      "\n **********************************",\
      )
model3.CalcDynamics(model3.k_path)

Y_hat = [(np.log(model3.y_path[t]) - np.log(model3.y_ss)) * 100 for t in range(8)]
L_hat = [(np.log(model3.l_path[t]) - np.log(model3.l_ss)) * 100 for t in range(8)] # This is not used in (b), but in (c)
I_hat = [(np.log(model3.x_path[t]) - np.log(model3.x_ss)) * 100 for t in range(8)]
C_hat = [(np.log(model3.c_path[t]) - np.log(model3.c_ss)) * 100 for t in range(8)]
RR    = [model3.r_path[t] * 100 for t in range(8)]

# Plot
x = range(8)
fig, ax = plt.subplots(2, 2, figsize=(8,8))

ax[0, 0].plot(x, Y_hat)
ax[0, 0].set_title('Output gap')
ax[1, 0].plot(x, C_hat)
ax[1, 0].set_title('Consumption gap')
ax[0, 1].plot(x, I_hat)
ax[0, 1].set_title('Investment gap')
ax[1, 1].plot(x, RR)
ax[1, 1].set_title('Real interest rate')

plt.savefig('Other_Variable_Responses_to_the_shock.png')

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

# Apply HP-filter to log Y, log L, log I, and log C
# The resulting cyclical components are log-difference between observed data
# and filtered trend, meaning they have the same unit as the counterparts 
# in the model
Y_hat_obs, _ = hpfilter(np.log(Y_obs), 1600)
L_hat_obs, _ = hpfilter(np.log(L_obs), 1600)
I_hat_obs, _ = hpfilter(np.log(I_obs), 1600)
C_hat_obs, _ = hpfilter(np.log(C_obs), 1600)


print(R_obs['2021'])
print(Y_hat_obs)
print(type(Y_obs))
print(type(R_obs))
print("\n")
