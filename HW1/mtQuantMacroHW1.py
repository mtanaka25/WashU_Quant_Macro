#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:51:54 2022

@author: tanapanda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Det_Neoclassical_Growth_Mdl:
    def __init__(self,
                 sigma = 2.000, # risk aversion
                 beta  = 0.960, # discount factor
                 theta = 0.330, # capital share in production function
                 delta = 0.081, # depreciation rate
                 A     = 0.592  # TFP
                 ):
        self.sigma = sigma
        self.beta  = beta
        self.theta = theta
        self.delta = delta
        self.A     = A

        
