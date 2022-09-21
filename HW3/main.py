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

tauchen = SIMModel()
tauchen.tauchen_discretize()
tauchen.run_simulation()

print('end')