#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]

N_TRAIN = 100;
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

## WITHOUT NORMARLIZATION
print('Regression without Variable Normalization:')
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

#initialize two dictionaries to store train and test error
deg=6
train_err = dict()
test_err = dict()

#calculate train and test error for each degree of polynominal 1- 6
for interator in range(1,deg+1):
    #calculate theta and w*    
    theta_train = a1.create_theta(x_train,interator)
    w = np.linalg.pinv(theta_train)*t_train

    y_train = np.transpose(w)*np.transpose(theta_train)
    t_train_error = t_train - np.transpose(y_train)
    rms_train_error = np.sqrt(np.mean(np.square(t_train_error)))

    theta_test = a1.create_theta(x_test,interator)
    y_test = np.transpose(w)*np.transpose(theta_test)
    t_test_error = t_test - np.transpose(y_test)
    rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))
    
    train_err[interator] = rms_train_error
    test_err[interator] = rms_test_error

# Produce a plot of results.
plt.plot([float(k) for k in train_err.keys()],[float(v) for v in train_err.values()])
plt.plot([float(k) for k in test_err.keys()],[float(v) for v in test_err.values()])
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
print('')

## WITH NORMARLIZATION
print('Regression with Variable Normalization:')
x_normalized = a1.normalize_data(x)
x_train = x_normalized[0:N_TRAIN,:]
x_test = x_normalized[N_TRAIN:,:]

#initialize two dictionaries to store train and test error
train_err = dict()
test_err = dict()

#calculate train and test error for each degree of polynominal 1- 6
for interator in range(1,deg+1):
    #calculate theta and w*
    theta_train = a1.create_theta(x_train,interator)
    w = np.linalg.pinv(theta_train)*t_train

    y_train = np.transpose(w)*np.transpose(theta_train)
    t_train_error = t_train - np.transpose(y_train)
    rms_train_error = np.sqrt(np.mean(np.square(t_train_error)))

    theta_test = a1.create_theta(x_test,interator)
    y_test = np.transpose(w)*np.transpose(theta_test)
    t_test_error = t_test - np.transpose(y_test)
    rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))
    
    train_err[interator] = rms_train_error
    test_err[interator] = rms_test_error

# Produce a plot of results.
plt.plot([float(k) for k in train_err.keys()],[float(v) for v in train_err.values()])
plt.plot([float(k) for k in test_err.keys()],[float(v) for v in test_err.values()])
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()