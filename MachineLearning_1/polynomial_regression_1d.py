# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:17:49 2016

@author: linhb
"""

#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
# Feature 8-15 will have index 7-14
x = values[:,7:15]

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# create matrix with theta from 1 to degree of polynominal
def create_theta(feature, deg):
    theta = np.ones(feature.shape[0], dtype=int)
    theta = np.reshape(theta,(feature.shape[0],1))
    for interator in range(1,deg+1):
        data_poly = np.apply_along_axis(np.power,0,feature,interator)
        theta = np.concatenate((theta,data_poly),1)
    return theta

#initialize two dictionaries to store train and test error
train_err = dict()
test_err = dict()

#calculate train and test error for each feature with polynominal degree = 0
for index in range(0,x.shape[1]):
    x_train_feature = x_train[:,index]
    x_test_feature = x_test[:,index]
    
    #calculate theta and w*
    theta_train = create_theta(x_train_feature,3)
    w = np.linalg.pinv(theta_train)*t_train

    y_train = np.transpose(w)*np.transpose(theta_train)
    t_train_error = t_train - np.transpose(y_train)
    rms_train_error = np.sqrt(np.mean(np.square(t_train_error)))

    theta_test = create_theta(x_test_feature,3)
    y_test = np.transpose(w)*np.transpose(theta_test)
    t_test_error = t_test - np.transpose(y_test)
    rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))
    
    train_err[8+index] = rms_train_error
    test_err[8+index] = rms_test_error


# Produce a plot of results.
plt.bar(np.arange(x.shape[1]), [float(v) for v in train_err.values()],0.33,
                 color='blue',
                 label='Train Error')
plt.bar(np.arange(x.shape[1])+0.33, [float(v) for v in test_err.values()],0.33,
                 color='green',
                 label='Test Error')
plt.xticks(np.arange(x.shape[1])+0.33,[('F'+ str(k)) for k in train_err.keys()])
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Single feature with polynominal degree = 3, no regularization')
plt.xlabel('Feature (F)')
plt.show()

execfile('visualize_1d.py')
