import numpy as np
import matplotlib.pyplot as plt

def fun(x, y):
  return 6 + 2*x**2 + 2*y**2

#Draw function z = 6 + 2*x**2 + 2*y**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

#Draw y = 8
fig = plt.figure('ro')
bx = fig.add_subplot(111, projection='3d')
ts = np.array([8 for x,y in zip(np.ravel(X), np.ravel(Y))])
T = ts.reshape(X.shape)
bx.plot_surface(X, Y, T)
bx.set_xlabel('x1')
bx.set_ylabel('x2')
bx.set_zlabel('y')
plt.show()

#Draw circle
fig=plt.figure(1)
plt.axis([-3,3,-3,3])
ax=fig.add_subplot(1,1,1)
circ=plt.Circle((0,0), radius=1, color='g', fill=False)
ax.add_patch(circ)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()