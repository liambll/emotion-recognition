#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()
features[10]


targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

def plot_fit(feature):
    print('Features: ' + str(feature+1) + ' - ' +features[feature])
    # Select a single feature.
    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]

    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

    # TO DO:: Put your regression estimate here in place of x_ev.
    theta_train = a1.create_theta(x_train,3)
    w = np.linalg.pinv(theta_train)*t_train
    theta_ev = a1.create_theta(np.transpose(np.asmatrix(x_ev)),3)
    y_ev = np.transpose(w)*np.transpose(theta_ev)

   
   # Evaluate regression on the linspace samples.
   #y_ev = np.random.random_sample(x_ev.shape)
   #y_ev = 100*np.sin(x_ev)


    plt.plot(x_train,t_train,'bo')
    plt.plot(x_test,t_test,'go')
    plt.plot(x_ev,np.transpose(y_ev),'r.-')
    plt.legend(['Training data','Test data','Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
    print('')

plot_fit(10)
plot_fit(11)
plot_fit(12)
