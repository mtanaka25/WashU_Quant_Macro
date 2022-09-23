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
# =========================================================================
# Settings
# =========================================================================
seed_to_use = 1234

SCF2019_file = 'rscfp2019.dta'
SCF2004_file = 'rscfp2004.dta'


# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro3 import SIMModel, SCF_hetero_income
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




# =========================================================================
# Question 1.
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question 1.             ",\
      "\n **********************************",\
      )
# -------------------------------------------------------------------------
# 1) Question 1 (a)-(e)
# -------------------------------------------------------------------------
# Create an instance
model_T = SIMModel()
# Discretize a la Tauchen
model_T.discretize(method='Tauchen')
# Run simulation
model_T.run_simulation(fname_header='Tauchen', fixed_seed=seed_to_use)

# -------------------------------------------------------------------------
# 2) Question 1 (f)-(g)
# -------------------------------------------------------------------------
# Create another instance
model_R = SIMModel()
# Discretize a la Rouwehhorst
model_R.discretize(method='Rouwenhorst')
# Run simulation (To compare with Tauchen, fix the seed)
model_R.run_simulation(fname_header='Rouwenhorst', fixed_seed=seed_to_use)

# -------------------------------------------------------------------------
# 3) Plot the results of (g)  
# -------------------------------------------------------------------------
# Gini coeficients by ages in each model
fig1 = plt.figure(figsize=(8, 6))
plt.plot(model_T.age_list, model_T.Gini_by_age, 
             color='blue', linestyle='dashed',
             label='Tauchen')
plt.plot(model_R.age_list, model_R.Gini_by_age, 
             color='red', label='Rouwehhorst')
fig1.savefig('Comparison_Gini.png', dpi=300)

# Lorenz curve for age 65 in each models
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


# =========================================================================
# Question 2.
# =========================================================================

SCF2019 = SCF_hetero_income(fname=SCF2019_file)
SCF2019.calc_Lorenz_for_three_groups(fname_header='SCF2019')
SCF2019.calc_Gini_index_by_age(fname_header='SCF2019')

SCF2004 = SCF_hetero_income(fname=SCF2004_file)
SCF2004.calc_Gini_index_by_age(fname_header='SCF2004')

fig3 = plt.figure(figsize=(8, 6))
plt.plot(model_T.age_list, model_T.Gini_by_age, 
             color='blue', linestyle='dashed',
             label='Tauchen')
plt.plot(model_R.age_list, model_R.Gini_by_age, 
             color='purple', label='Rouwehhorst')
plt.plot(model_R.age_list, SCF2019.Gini_by_age, 
             color='red', label='SCF 2019')
plt.legend(frameon = False)
fig3.savefig('Comparison_Gini_SCF2019.png', dpi=300)

fig4 = plt.figure(figsize=(8, 6))
plt.plot(model_T.age_list, model_T.Gini_by_age, 
             color='blue', linestyle='dashed',
             label='Tauchen')
plt.plot(model_R.age_list, model_R.Gini_by_age, 
             color='purple', label='Rouwehhorst')
plt.plot(model_R.age_list, SCF2004.Gini_by_age, 
             color='red', label='SCF 2004')
plt.legend(frameon = False)
fig4.savefig('Comparison_Gini_SCF2004.png', dpi=300)

param0 = [model_T.rho, model_T.sig_eps, model_T.sig_y20]
optimal_param = SCF2004.recalibrate_AR_params(param0=param0)
print('\n rho = {:.4f}'.format(optimal_param[0]))
print('\n var_eps = {:.4f}'.format(optimal_param[1]**2))
print('\n var_y20 = {:.4f}'.format(optimal_param[2]**2))

model_T_recalib = SIMModel(rho=optimal_param[0],
                           var_eps=optimal_param[1]**2,
                           var_y20=optimal_param[2]**2)
model_T_recalib.discretize(method='tauchen')
model_T_recalib.run_simulation(is_plot=False)

fig5 = plt.figure(figsize=(8, 6))
plt.plot(model_T.age_list, model_T.Gini_by_age, 
             color='blue', linestyle='dashed',
             label='Original')
plt.plot(model_T_recalib.age_list, model_T_recalib.Gini_by_age, 
             color='purple', label='Recalibration')
plt.plot(model_R.age_list, SCF2004.Gini_by_age, 
             color='red', label='SCF 2004')
plt.legend(frameon = False)
fig5.savefig('Result_of_recalibration.png', dpi=300)