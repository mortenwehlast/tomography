#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:41:52 2018

@author: Morten
"""
import numpy as np
import matplotlib as plt
from paralleltomo import paralleltomo



### Matrix A
# A pixel hit is denoted by the length of the beam going trough the pixel
# Rows are each of the p rays for q angles
# Columns are pixels hit
# Thus
# A has dimensions p*q X n^2

# The paralleltomo output is assumed to be NxN (pixel side length = 1 unit)
# We neeed to scale to LxL (pixel side length = Delta L)


### Models

# N and L provided from exercise text

N = 64
L = 1
deltaL = L/N

### 2.1 Unique Solution



# Configuration 1a and 1b
theta1_a = np.asarray(np.arange(1.5, 97.5, 1.5))
theta1_b = np.asarray(np.arange(2.5, 162.5, 2.5))
n_rays1  = int(N)
dist1    = n_rays1 - 1

# Configuration 2a and 2b
theta2_a = np.asarray(np.arange(0.75, 96.75, 0.75))
theta2_b = np.asarray(np.arange(1.25, 161.25, 1.25))
n_rays2  = int(np.floor(N/2))
dist2    = n_rays2 - 1


### Calculate A

# Configuration 1
[A_1a, theta1_a2, nrays1a, dist1a] = paralleltomo(N, theta1_a, n_rays1, dist1)
[A_1b, theta1_b2, nrays1b, dist1b] = paralleltomo(N, theta1_b, n_rays1, dist1)

# Configuration 2
[A_2a, theta2_a2, nrays2a, dist2a] = paralleltomo(N, theta2_a, n_rays2, dist2)
[A_2b, theta2_b2, nrays2b, dist2b] = paralleltomo(N, theta2_b, n_rays2, dist2)

# Resize slices from NxN to LxL
A_1a, A_1b, A_2a, A_2b = deltaL * A_1a, deltaL * A_1b, 2*deltaL * A_2a, 2*deltaL * A_2b



