#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #3 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW3.py

..........................................................................
Create Sep 20, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

from mtQuantMacro3 import SIMModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

model_T = SIMModel()
model_T.discretize(method='Tauchen')
model_T.run_simulation(fname_header='Tauchen')

model_R = SIMModel()
model_R.discretize(method='Rouwenhorst')
model_R.run_simulation(fname_header='Rouwenhorst')

fig1 = plt.figure(figsize=(8, 6))
plt.plot(model_T.age_list, model_T.Gini_by_age, 
             color='blue', linestyle='dashed',
             label='Tauchen')
plt.plot(model_R.age_list, model_R.Gini_by_age, 
             color='red', label='Rouwehhorst')
fig1.savefig('Comparison_Gini.png', dpi=300)

fig2 = plt.figure(figsize=(8, 6))
plt.plot(model_T.Lorenz_by_age[-1][0], 
         model_T.Lorenz_by_age[-1][0], 
         '-', linewidth = 0.5, color = 'green', label='perfect equality')
plt.plot(model_T.Lorenz_by_age[-1][0], 
         model_T.Lorenz_by_age[-1][1], 
         color='blue', linestyle='dashed',
         label='Tauchen')
plt.plot(model_R.Lorenz_by_age[-1][0], 
         model_R.Lorenz_by_age[-1][1], 
         color='red', label='Rouwehhorst')
plt.hlines(0, 0, 1, color='black', linewidth = 0.5, 
        linestyles='--', label='perfect inequality')
plt.vlines(1, 0, 1, color='black', linewidth = 0.5, 
        linestyles='--')
plt.legend(frameon = False)
fig2.savefig('Comparison_Lorenz_Y65.png', dpi=300)
print('end')