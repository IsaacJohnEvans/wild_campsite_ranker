from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from math import e, pi, sqrt


def normal_func(x,y,mean,var):
    return np.exp(-0.5*(((x**2 + y**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)

var = 10
mean = 10
x = np.outer(np.linspace(-50, 50, 100), np.ones(100))
y = x.copy().T
z1 = normal_func(x,y,mean,var)

var = 5
mean = 5
x = np.outer(np.linspace(-50, 50, 100)+30, np.ones(100))
y = x.copy().T
z2 = normal_func(x,y,mean,var)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z1+z2, cmap ='viridis', edgecolor ='green')

plt.show()