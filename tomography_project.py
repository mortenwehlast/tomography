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

# Create b for dimensions in noise creation. Pick whichever matrix
A = A_1a
b = np.dot(A,x)


# Numerical experiments
# Create matrix with matrices with full rank + condition numbers
matrices = [A_1a, A_1b, A_2b]
cond_num = [cond_A_1a, cond_A_1b, cond_A_2b]

# Pre-calculate inverse of matrices
matrices_inv = [np.linalg.inv(A_1a), np.linalg.inv(A_1b), np.linalg.inv(A_2b)]


# Allocate memory for relative error measurements and noise
# 10 tests, 6 noise levels (magnitude of 10 between steps)
n_matrix = len(matrices)
n_levels = 10
n_tests = 10
rel_error_mat = np.zeros([n_matrix, n_levels])
max_error_mat = np.zeros([n_matrix, n_levels])


# Create noise for matrices
# Set seed
np.random.seed(146782)
noise_mat = np.zeros([4096,n_tests,n_levels])

# Noise levels (for plot)
noise_levels = np.zeros([n_levels, 1])

for i in range(0, n_levels):
    noiselevel = 10**(-(i))
    noise_levels[i] = noiselevel
    
    for j in range(0, n_tests):
        noise_mat[:,j,i] = (noiselevel*np.random.normal(0,1,b.shape)).squeeze()
      


# Calculate errors for matrices. Error calcalutated as mean over all 10 tests

for i in range(0, n_levels):
    
    #Loop over matrices
    for j in range(0, n_matrix):
        print(j,i)
        A           = matrices[j]
        A_inv       = matrices_inv[j]
        b           = np.dot(A,x)
        kappa       = cond_num[j]
        errors      = np.zeros([n_tests, 1])
        upper_bound = np.zeros([n_tests, 1])
        print(b.shape, noise_mat[:,2,i].shape)
        
        # Calc errors for each noise vector at level i
        for k in range(0, n_tests):
            btilde         = b.squeeze() + noise_mat[:,k,i]
            xtilde         = np.dot(A_inv, btilde)
            errors[k]      = np.linalg.norm(x-xtilde)/np.linalg.norm(x)
            #upper_bound[k] = kappa*np.linalg.norm(b-btilde)/np.linalg.norm(b)
            
        #max_error_mat[j,i] = np.mean(upper_bound)
        rel_error_mat[j,i] = np.mean(errors)

        


### Plots

# Logarithmic scaling of noise levels
log_noise = np.log10(noise_levels)
log_error = np.log10(rel_error_mat)

# Relative error as function of noise level
plt.figure(1)
plt.plot(log_noise, log_error[0,:], 'r*-', label = 'Configuration 1a')
plt.plot(log_noise, log_error[1,:], 'b^-', label = 'Configuration 1b')
plt.plot(log_noise, log_error[2,:], 'gs-', label = 'Configuration 2b')
plt.legend(loc = 'upper left')
plt.title('Relative error as a function of noise magnitude')
plt.xlabel('Noise level (log10)')
plt.ylabel('Relative error (log10)')
plt.show()

# Shepp logan reconstruction for 3 levels (10^(-6), 10^(-5), 10^(-4))
# Chosen based on graph from before

plot_num = 1
fig = plt.figure()
for i in range(0,3):
    for j in range(0,3):
        mat = j
        b = np.dot(matrices[mat], x)
        btilde = b.squeeze() + noise_mat[:,0, 5 + i]
        xtilde = np.dot(matrices_inv[mat], btilde)
        plt.subplot(3,3, plot_num)
        plt.imshow(np.reshape(xtilde, (N, N)))
        plt.axis('off')
        plot_num += 1
plt.show()


# Plot the reconstruction
plt.imshow(np.reshape(xtilde, (N, N)))

