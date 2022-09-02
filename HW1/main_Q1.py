#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:51:54 2022

@author: tanapanda
"""

from mtQuantMacroHW1 import Det_NCG_Mdl

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
k_1, l_1 = model1.DoBisection_k1_l1()
