#coding utf8
#%%
import matplotlib.pyplot as plt
import numpy as np
from math import e, pi
#%%
var = 0.1
mean = 3
n_points = 1000
min_point = -5
max_point = 5
x = np.outer(np.linspace(min_point, max_point, n_points), np.ones(n_points))
y = x.copy().T
x_cen = 0
y_cen = 0
z = np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
plt.show()

#%%
x = 0
y = 0 
x_cen = 0
y_cen = 0
mean = 1

np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)
np.tanh(np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5))
