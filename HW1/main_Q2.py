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
model3.DoExtendedPath(k_path_guess, k_min = model3.k_ss * 0.5, k_max = k_ss_new * 1.1, GraphicName='Econ5725_HW01_Q2a.png')


# -------------------------------------------------------------------------
# (b) Calculate the other variables' dynamics
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (b)            ",\
      "\n **********************************",\
      )
model3.CalcDynamics(model3.k_path)

Y_hat = [(np.log(model3.y_path[t]) - np.log(model3.y_ss)) * 100 for t in range(9)]
L_hat = [(np.log(model3.l_path[t]) - np.log(model3.l_ss)) * 100 for t in range(9)]
I_hat = [(np.log(model3.x_path[t]) - np.log(model3.x_ss)) * 100 for t in range(9)]
C_hat = [(np.log(model3.c_path[t]) - np.log(model3.c_ss)) * 100 for t in range(9)]
RR    = [model3.r_path[t] * 100 for t in range(9)]

# -------------------------------------------------------------------------
# (c) Implement extended path to derive the dynamics of k and l
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 2. (c)            ",\
      "\n **********************************",\
      )
# Connect to FRED
# Use your own api key. You can apply the key on the website of FRED (for free).
fred = Fred(api_key = '###################') 


# Obtain the necessary data from FRED
Y_obs = fred.get_series('GDPC1')
L_obs = fred.get_series('PAYEMS')
I_obs = fred.get_series('GPDIC1')
C_obs = fred.get_series('PCECC96')

# For employment data, convert from monthly to quarterly.
L_obs = L_obs.resample('Q', convention='end').mean()

# Apply HP-filter to log Y, log L, log I, and log C
# The resulting cyclical components are log-difference between observed data
# and filtered trend, meaning they have the same unit as the counterparts 
# in the model
Y_hat_obs, _ = hpfilter(np.log(Y_obs), 1600)
L_hat_obs, _ = hpfilter(np.log(L_obs), 1600)
I_hat_obs, _ = hpfilter(np.log(I_obs), 1600)
C_hat_obs, _ = hpfilter(np.log(C_obs), 1600)

# Pick up the data from 2019Q4 to 2021Q4
Y_hat_obs = (Y_hat_obs.truncate(before = '2019-10-1', after = '2021-12-31')) * 100
L_hat_obs = (L_hat_obs.truncate(before = '2019-10-1', after = '2021-12-31')) * 100
I_hat_obs = (I_hat_obs.truncate(before = '2019-10-1', after = '2021-12-31')) * 100
C_hat_obs = (C_hat_obs.truncate(before = '2019-10-1', after = '2021-12-31')) * 100


# The data on real interest rates is available on World Bank database
# only on annual basis. So, following the explanation about data construction
# on the World Bank database, I am calculating quarterly data on US real
# interest rate by myself. 
# The necessary data (prime loan rate and GDP deflator) are obtained from FRED.
NR_obs = fred.get_series('MPRIME', observation_start='2019-10-01', observation_end='2021-12-31')
Pi_obs = fred.get_series('A191RI1Q225SBEA', observation_start='2019-10-01', observation_end='2021-12-31')

# For nominal interest rate (prime loan rate), convert from monthly to quarterly.
NR_obs = NR_obs.resample('Q', convention='end').last()

# Reset the index
NR_obs = pd.Series(NR_obs.tolist(), index = Pi_obs.index)

# Real rate = nominal rate - inflation rate
RR_obs = NR_obs - Pi_obs



# Plot
x      = Y_hat_obs.index
xLabel = ['19Q4', '20Q1', 'Q2', 'Q3', 'Q4', '21Q1', 'Q2', 'Q3', 'Q4']
fig, ax = plt.subplots(3, 2, figsize=(10,12))

ax[0, 0].plot(x, Y_hat    , label='model')
ax[0, 0].plot(x, Y_hat_obs, label='data' )
ax[0, 0].set_xticklabels(xLabel)
ax[0, 0].set_title('Output')
ax[0, 0].legend(frameon=False)

ax[1, 0].plot(x, C_hat    , label='model')
ax[1, 0].plot(x, C_hat_obs, label='data' )
ax[1, 0].set_xticklabels(xLabel)
ax[1, 0].set_title('Consumption')
ax[1, 0].legend(frameon=False)

ax[0, 1].plot(x, I_hat    , label='model')
ax[0, 1].plot(x, I_hat_obs, label='data' )
ax[0, 1].set_xticklabels(xLabel)
ax[0, 1].set_title('Investment')
ax[0, 1].legend(frameon=False)

ax[1, 1].plot(x, L_hat    , label='model')
ax[1, 1].plot(x, L_hat_obs, label='data' )
ax[1, 1].set_xticklabels(xLabel)
ax[1, 1].set_title('Employment')
ax[1, 1].legend(frameon=False)

ax[2, 0].plot(x, RR    , label='model')
ax[2, 0].plot(x, RR_obs, label='data' )
ax[2, 0].set_xticklabels(xLabel)
ax[2, 0].set_title('Real interest rate')
ax[2, 0].legend(frameon=False)

ax[2, 1].axis("off")

plt.savefig('Econ5725_HW01_Q2bc.png')