# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:02:26 2016

@author: linhb
"""
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

# sigmoid with u, s
def sigmoid(x,u,s):
    return 1/(1+np.exp((u-x)/s))

# create theta with sigmoid function
def create_theta_sigmoid(feature, u, s):
    data_poly = np.apply_along_axis(sigmoid,0,feature,u,s)
    return data_poly

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
# Feature 9 - GNI Per capital
x = values[:,10]

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

print('Regression with Sigmoid basic function on Feature 11 - GNI Per Capital')
#calculate theta and w*
u1 = 100
u2 = 10000
s = 2000
theta_train = np.ones(x_train.shape[0], dtype=int)
theta_train = np.reshape(theta_train,(x_train.shape[0],1))
theta_train = np.concatenate((theta_train,create_theta_sigmoid(x_train,u1,s)),1)
theta_train = np.concatenate((theta_train,create_theta_sigmoid(x_train,u2,s)),1)
w = np.linalg.pinv(theta_train)*t_train

y_train = np.transpose(w)*np.transpose(theta_train)
t_train_error = t_train - np.transpose(y_train)
rms_train_error = np.sqrt(np.mean(np.square(t_train_error)))

theta_test = np.ones(x_test.shape[0], dtype=int)
theta_test = np.reshape(theta_test,(x_test.shape[0],1))
theta_test = np.concatenate((theta_test,create_theta_sigmoid(x_test,u1,s)),1)
theta_test = np.concatenate((theta_test,create_theta_sigmoid(x_test,u2,s)),1)
y_test = np.transpose(w)*np.transpose(theta_test)
t_test_error = t_test - np.transpose(y_test)
rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))

# Produce a plot of fit
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.transpose(np.asmatrix(x_ev))
# TO DO:: Put your regression estimate here in place of x_ev.
theta_ev = np.ones(x_ev.shape[0], dtype=int)
theta_ev = np.reshape(theta_ev,(x_ev.shape[0],1))
theta_ev = np.concatenate((theta_ev,create_theta_sigmoid(x_ev,u1,s)),1)
theta_ev = np.concatenate((theta_ev,create_theta_sigmoid(x_ev,u2,s)),1)
y_ev = np.transpose(w)*np.transpose(theta_ev)

# Evaluate regression on the linspace samples.
#y_ev = np.random.random_sample(x_ev.shape)
#y_ev = 100*np.sin(x_ev)


plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.plot(x_ev,np.transpose(y_ev),'r.-')
plt.legend(['Training data','Test data','Learned Function'])
plt.title('A visualization of a regression estimate using random outputs')
plt.show()

print('Train Error: %f' % rms_train_error)
print('Test Error: %f' % rms_test_error) 
print('')


