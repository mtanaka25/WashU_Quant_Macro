#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
main.py

is main code for the assignment #4 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW3.py

..........................................................................
Create Sep 27, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

# =========================================================================
# Settings
# =========================================================================



# =========================================================================
# Load packages
# =========================================================================
from mtQuantMacro4 import IFP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# =========================================================================
# Question (a).
# =========================================================================
print("\n",
      "\n **********************************",\
      "\n           Question (a)             ",\
      "\n **********************************",\
      )
# Create an instance
model_a = IFP()
# Discretize the income process by Rouwehhorst method
model_a.discretize(method='Rouwenhorst')

model_a.policy_function_iter()
