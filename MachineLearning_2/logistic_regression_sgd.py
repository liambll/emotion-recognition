#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]


# Load data.
data = np.genfromtxt('data.txt')
#Shuffle data
np.random.shuffle(data)
# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]
        
plt.figure()
legend = []
for eta in etas:
    legend.append(str(eta))
    # Initialize w.
    w = np.array([0.1, 0, 0])

    # Error values over all iterations.
    e_all = []
    for iter in range (0,max_iter):
    
        # Compute output using current w on training record ran.
        #y = sps.expit(np.dot(X,w))
        

        # Gradient of the error using stochastic approach
        for i in np.arange(0,data.shape[0]):
            # Compute gradient using training record i
            y = sps.expit(np.dot(X[i],w))
            grad_e = np.multiply((y - t[i]), X[i].T)
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w
            w = w - eta*grad_e
            #print 'inter {0:d}, w={1}'.format(i, w.T)

        # Compute output using current w on all training records.
        y = sps.expit(np.dot(X,w))
            
        # e is the error, negative log-likelihood (Eqn 4.90)
        # e-24 is added in to avoid log 0
        e = -np.mean(np.multiply(t,np.log(y+1e-24)) + np.multiply((1-t),np.log(1-y+1e-24)))
        #e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
        # Add this error to the end of error vector.
        e_all.append(e)
        
        # Stop iterating if error doesn't change more than tol.
        if iter>0:
            if np.absolute(e-e_all[iter-1]) < tol:
                break
    #print some info
    #print 'epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T)  
    plt.plot(e_all)
    
    #iteration.append(iter)
    #minfunction.append(e)
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with Stochastics Gradient Descent')
plt.xlabel('Epoch')
plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5)
plt.show()

# Plot error and iterations over learning rate
#fig, ax1 = plt.subplots()
#ax1.plot(etas, minfunction, 'b-')
#ax1.set_xlabel('Learning Rate')
#ax1.set_ylabel('Negative log likelihood', color='b')
#ax1.set_title('Negative log likelihood and Iteration over Learning Rate')

#ax2 = ax1.twinx()
#ax2.plot(etas, iteration, 'r-')
#ax2.set_ylabel('Iteration', color='r')
#plt.show()