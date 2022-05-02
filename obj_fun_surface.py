from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from math import e, pi, sqrt
from OSGridConverter import latlong2grid


def sinusoidal_func(x, y, dist, h):
    return -h * np.cos(((x**2 + y**2) ** 0.5) / dist) + h


def normal_func(x, y, mean, var):
    return np.exp(-0.5 * (((x**2 + y**2) ** 0.5 - mean) / var) ** 2) / (
        var * (2 * pi) ** 0.5
    )
    # x = np.where((x**2 + y**2)**0.5 > 2*pi, 0, x)
    # y = np.where((x ** 2 + y ** 2) ** 0.5 > 2*pi, 0, y)
    # return -np.cos((x**2 + y**2)**0.5)


def step_func(x, y):
    return -5


import json

with open("bristol_test_data/bristol_pubs_points.geojson", "r") as read_file:
    pub_dict = json.load(read_file)


def get_point_coords(data_dict):
    coords = []
    grid_refs = []
    for i in range(len(data_dict["features"])):
        coords.append(
            list(reversed(data_dict["features"][i]["geometry"]["coordinates"]))
        )
        grid_ref = latlong2grid(coords[i][0], coords[i][1])
        grid_refs.append([grid_ref.E, grid_ref.N])
    return coords, grid_refs


pub_coords, pub_gridrefs = get_point_coords(pub_dict)
print(pub_gridrefs[:1])

res = 1000
x = np.outer(np.linspace(360000, 368000, res, endpoint=False), np.ones(res))
y = np.outer(np.linspace(170000, 178000, res, endpoint=False), np.ones(res)).T
z = np.zeros_like(x)
for pub in pub_gridrefs:
    z += normal_func(x - pub[0], y - pub[1], 300, 300)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(x, y, z, cmap="viridis", edgecolor="green")

plt.show()
