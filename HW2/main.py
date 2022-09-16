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
GHH_Qa = GHHModel()
GHH_Qa.value_func_iter()


# -------------------------------------------------------------------------
# (b) Solve with 1000 grid points and without any speed-up method
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (b)            ",\
      "\n **********************************",\
      )
GHH_Qb = GHHModel(nGrids=1000)
GHH_Qb.value_func_iter()

# -------------------------------------------------------------------------
# (c) Solve with 1000 grid points and with speed-up methods
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (c)            ",\
      "\n **********************************",\
      )
GHH_Qc_mono = GHHModel(nGrids=1000)
GHH_Qc_mono.value_func_iter(is_monotone=True)

GHH_Qc_concave = GHHModel(nGrids=1000)
GHH_Qc_concave.value_func_iter(is_concave=True)

GHH_Qc_both = GHHModel(nGrids=1000)
GHH_Qc_both.value_func_iter(is_monotone=True, is_concave=True)

# -------------------------------------------------------------------------
# (d) Solve with 1000 grid points and with Howard method
# -------------------------------------------------------------------------
print("\n",
      "\n **********************************",\
      "\n        Question 1. (d)            ",\
      "\n **********************************",\
      )
GHH_Qd_10 = GHHModel(nGrids=1000)
GHH_Qd_10.value_func_iter(is_monotone=True, 
                          is_concave=True,
                          is_modified_policy_iter=True,
                          n_h=10)

GHH_Qd_25 = GHHModel(nGrids=1000)
GHH_Qd_25.value_func_iter(is_monotone=True, 
                          is_concave=True,
                          is_modified_policy_iter=True,
                          n_h=25)

GHH_Qd_100 = GHHModel(nGrids=1000)
GHH_Qd_100.value_func_iter(is_monotone=True, 
                          is_concave=True,
                          is_modified_policy_iter=True,
                          n_h=100)