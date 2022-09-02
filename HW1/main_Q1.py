#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:51:54 2022

@author: tanapanda
"""

from mtQuantMacroHW1 import Det_NCG_Mdl
import numpy as np

# =========================================================================
# 1.
# =========================================================================

# Generate a NCG model instance
model1 = Det_NCG_Mdl()

# (a) Calculate the steady state
model1.SteadyState()

# (b-1) Calculate l_0 corresponding to the given k_0
l_0 = model1.Calc_l(k_t = model1.k_0, isQ1b = True)
model1.l_0 = l_0

# (b-2) Implement bisection to find k_1 and l_1
k_2 = 0.82
k_1, l_1 = model1.DoBisection(k_t = model1.k_0, k_tp2 = k_2)

# (b-3) Implement Newton method to find k_1 and l_1
k_1_Newton, l_1_Newton = model1.DoNewton(k_t = model1.k_0, k_tp2 = k_2)


# (c) Implement extended path to derive the dynamics of k 
k_path_guess = [model1.k_0 + (model1.k_ss - model1.k_0) * i / 151 for i in range(151)]
model1.DoExtendedPath(k_path_guess)

# (d) Calculate the other variables' dynamics 
model1.CalcDynamics(model1.k_path)
