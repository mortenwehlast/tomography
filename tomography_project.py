#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:41:52 2018

@author: Morten
"""
import numpy as np
import matplotlib.pyplot as plt
from paralleltomo import paralleltomo
import phantom


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

# Calculate Condition numbers
cond_A_1a = np.linalg.cond(A_1a)
cond_A_1b = np.linalg.cond(A_1b)
cond_A_2a = np.linalg.cond(A_2a)
cond_A_2b = np.linalg.cond(A_2b)

# Calculate rank of matrices
rank_A_1a = np.linalg.matrix_rank(A_1a)
rank_A_1b = np.linalg.matrix_rank(A_1b)
rank_A_2a = np.linalg.matrix_rank(A_2a)
rank_A_2b = np.linalg.matrix_rank(A_2b)



### Verification

# Create shepp logan phantom
X = phantom.shepp2d(N)
x = np.reshape(X, (N*N, 1))
A = A_1b
b = np.dot(A,x)

# Numerical experiments
# Create matrix with matrices with full rank + condition numbers
matrices = [A_1a, A_1b, A_2b]
cond_num = [cond_A_1a, cond_A_1b, cond_A_2b]

# Pre-calculate inverse of matrices
matrices_inv = np.linalg.inv(matrices)


# Allocate memory for relative error measurements and noise
# 10 tests, 6 noise levels (magnitude of 10 between steps)
n_matrix = len(matrices)
n_levels = 6
n_tests = 10
rel_error_mat = np.zeros([n_matrix, n_levels])
max_error_mat = np.zeros([n_matrix, n_levels])


# Create noise for matrices
# Set seed
np.random.seed(142)
noise_mat = np.zeros([4096,n_tests,n_levels])

for i in range(0, n_levels):
    noiselevel = 10**(-(i+1))
    
    for j in range(0, n_tests):
        noise_mat[:,j,i] = (noiselevel*np.random.normal(0,1,b.shape)).squeeze()
      


# Calculate errors for matrices. Error calcalutated as mean over all 10 tests

for i in range(0, n_levels):
    
    #Loop over matrices
    for j in range(0, n_matrix):
        A           = matrices[j]
        A_inv       = matrices_inv[j]
        b           = np.dot(A,x)
        kappa       = cond_num[j]
        errors      = np.zeros([n_tests, 1])
        upper_bound = np.zeros([n_tests, 1])
        
        # Calc errors for each noise vector at level i
        for k in range(0, n_tests):
            btilde         = b + noise_mat[:,k,i]
            xtilde         = np.dot(A_inv, btilde)
            errors[k]      = np.linalg.norm(x-xtilde)/np.linalg.norm(x)
            #upper_bound[k] = kappa*np.linalg.norm(b-btilde)/np.linalg.norm(b)
            
        #max_error_mat[j,i] = np.mean(upper_bound)
        rel_error_mat[j,i] = np.mean(errors)

        

#for i in range(0, n_matrix):
#    # Assign matrix to be tested
#    A = matrices[i]
#    b = np.dot(A,x)
#    kappa = np.linalg.cond(A)
#    
#    for j in range(0, n_levels):
#        noiselevel = 10**(-(j+1))
#        print(noiselevel)
#        btilde = b + noiselevel*np.random.normal(0,1,b.shape)
#        xtilde = np.linalg.solve(A, btilde)
#        
#        # add relative error and max possible error to matrices
#        max_error_mat[i,j] = kappa*np.linalg.norm(b-btilde)/np.linalg.norm(b)
#        rel_error_mat[i,j] = np.linalg.norm(x-xtilde)/np.linalg.norm(x)




# Plot the reconstruction
plt.imshow(np.reshape(xtilde, (N, N)))


### Numerical experiments