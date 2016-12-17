# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:17:49 2016

@author: linhb
"""

#!/usr/bin/env python

#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN=100
x = x[0:N_TRAIN,:]
targets = targets[0:N_TRAIN]

#initialize two dictionaries to store train and test error
validation_err_avg = dict()

#calculate train and test error for each lamda
def regression_reg(lamda, deg):
    #split train-test for 10 fold cross validation
    validation_err = 0
    bucket = 10 #size of validation set
    for interator in range(0,10):
        x_val = x[interator*bucket:(interator+1)*bucket,:]
        t_val = targets[interator*bucket:(interator+1)*bucket]
        x_train = np.concatenate((x[0:interator*bucket,:],x[(interator+1)*bucket:,:]),0)

        t_train = np.concatenate((targets[0:interator*bucket],targets[(interator+1)*bucket:]),0)

        #calculate theta and w*
        theta_train = a1.create_theta(x_train,deg)  
        w = np.linalg.inv(lamda*np.identity(theta_train.shape[1]) + np.transpose(theta_train).dot(theta_train)) \
        .dot(np.transpose(theta_train)).dot(t_train)

        theta_val = a1.create_theta(x_val,deg)
        y_val = np.transpose(w)*np.transpose(theta_val)
        t_val_error = t_val - np.transpose(y_val)
        rms_val_error = np.sqrt(np.mean(np.square(t_val_error)))
        
        validation_err += rms_val_error
    validation_err_avg[lamda] = validation_err/10

regression_reg(0.000001,2) #replace lambda=0 with lambda=10^-5 for plotting purpose.
regression_reg(0.01,2)
regression_reg(0.1,2)
regression_reg(1,2)
regression_reg(10,2)
regression_reg(100,2)
regression_reg(1000,2)
regression_reg(10000,2)

# Produce a plot of results.
label = sorted(validation_err_avg.keys())
error = []
for key in label:
    error.append(validation_err_avg[key])
    
plt.semilogx(label,error)
plt.ylabel('Average RMS')
plt.legend(['Average Validation error'])
plt.title('Fit with polynomial degree = 2, regularization with 10-fold cross validation')
plt.xlabel('lambda on log scale \n (lambda=10^-5 represents lambda=0 closely in terms of validation error)')
plt.show()