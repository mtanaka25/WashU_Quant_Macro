#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_Q1.py

is main code for the assignment #1 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW1.py

...............................................................................
Create Sep 1st, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

from mtQuantMacroHW1 import Det_NCG_Mdl
import matplotlib.pyplot as plt

# =========================================================================
# 1.
# =========================================================================
# Setting
T = 150

# Generate a NCG model instance
model1 = Det_NCG_Mdl(T = T)

# -------------------------------------------------------------------------
# (a) Calculate the steady state
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (a)            ",\
      "\n **********************************",\
      )
model1.SteadyState()

# -------------------------------------------------------------------------
# (b-1) Calculate l_0 corresponding to the given k_0
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (b)            ",\
      "\n **********************************",\
      )

l_0 = model1.Calc_l(k_t = model1.k_0)
model1.l_0 = l_0

print("\n",
      "\n  l_0 = {:.4f}".format(l_0)
      )

# -------------------------------------------------------------------------
# (b-2) Implement bisection to find k_1 and l_1
# -------------------------------------------------------------------------
k_2 = 0.82
k_1, l_1 = model1.DoBisection(k_t = model1.k_0, k_tp2 = k_2)

# (b-3) Implement Newton method to find k_1 and l_1
k_1_Newton, l_1_Newton = model1.DoNewton(k_t = model1.k_0, k_tp2 = k_2)

# -------------------------------------------------------------------------
# (c) Implement extended path to derive the dynamics of k and l
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (c)            ",\
      "\n **********************************",\
      )
# Initial guess: linear transition
k_path_guess = [model1.k_0 + (model1.k_ss - model1.k_0) * i / (T + 1) for i in range(T + 1)]
model1.DoExtendedPath(k_path_guess, k_max = model1.k_ss * 1.1, GraphicName='ExtendedPath_sigma2.png')

# -------------------------------------------------------------------------
# (d) Redo with the economy where sigma = 4
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (d)            ",\
      "\n **********************************",\
      )
model2 = Det_NCG_Mdl(sigma = 4.0)

model2.SteadyState()

k_path_guess = [model2.k_0 + (model2.k_ss - model2.k_0) * i / (T + 1) for i in range(T + 1)]
model2.DoExtendedPath(k_path_guess, k_max = model2.k_ss * 1.1,  GraphicName='ExtendedPath_sigma4.png')

# -------------------------------------------------------------------------
# (e) Calculate the other variables' dynamics
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (e)            ",\
      "\n **********************************",\
      )
model1.CalcDynamics(model1.k_path)
model2.CalcDynamics(model2.k_path)

# Plot
x = range(51)
fig, ax = plt.subplots(2, 2, figsize=(8,8))

ax[0, 0].plot(x, model1.y_path[0:51], label='sigma = {:.1f}'.format(model1.sigma))
ax[0, 0].plot(x, model2.y_path[0:51], label='sigma = {:.1f}'.format(model2.sigma))
ax[0, 0].set_title('Output')
ax[0, 0].legend(frameon=False)

ax[1, 0].plot(x, model1.c_path[0:51], label='sigma = {:.1f}'.format(model1.sigma))
ax[1, 0].plot(x, model2.c_path[0:51], label='sigma = {:.1f}'.format(model2.sigma))
ax[1, 0].set_title('Consumption')
ax[1, 0].legend(frameon=False)


ax[0, 1].plot(x, model1.x_path[0:51], label='sigma = {:.1f}'.format(model1.sigma))
ax[0, 1].plot(x, model2.x_path[0:51], label='sigma = {:.1f}'.format(model2.sigma))
ax[0, 1].set_title('Investment')
ax[0, 1].legend(frameon=False)


ax[1, 1].plot(x, model1.r_path[0:51], label='sigma = {:.1f}'.format(model1.sigma))
ax[1, 1].plot(x, model2.r_path[0:51], label='sigma = {:.1f}'.format(model2.sigma))
ax[1, 1].set_title('Interest rate')
ax[1, 1].legend(frameon=False)

plt.savefig('Other_Variable_Dynamics.png')
