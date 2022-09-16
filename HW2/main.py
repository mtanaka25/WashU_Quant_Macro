#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

is main code for the assignment #2 of Quantitative Macroeconomic Theory
at Washington University in St. Louis.

Requirement:
      mtQuantMacroHW2.py

..........................................................................
Create Sep 14, 2022 (Masaki Tanaka, Washington University in St. Louis)

"""

from mtQuantMacroHW2 import GHHModel

# =========================================================================
# Setting
# =========================================================================



# -------------------------------------------------------------------------
# (a) Solve with 100 grid points and without any speed-up method
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (a)            ",\
      "\n **********************************",\
      )

# Generate a GHH model instance
GHH_Qa = GHHModel()
GHH_Qa.value_func_iter()
