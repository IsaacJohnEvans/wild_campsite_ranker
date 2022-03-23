#coding utf8
#%%
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from math import e, pi
import matplotlib.path as mpltPath
import json
from OSGridConverter import latlong2grid
#%%
var = 1
mean = 3
n_points = 1000
min_point = -5
max_point = 5
x = np.outer(np.linspace(min_point, max_point, n_points), np.ones(n_points))
y = x.copy().T
x_cen = 0
y_cen = 0
z = np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)
fig = plt.figure(1)
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
#%%
with open('bristol_test_data/bristol_pubs_multipolygons.geojson', "r") as read_file:
    pub_poly_dict = json.load(read_file)

def get_poly_coords(poly_dict):
    poly_coords = []
    poly_grid_refs = []
    for i in range(len(poly_dict['features'])):
        j_list = []
        for j in range(len(poly_dict['features'][i]['geometry']['coordinates'])):
            full_poly = []
            full_grid_poly = []
            for k in range(len(poly_dict['features'][i]['geometry']['coordinates'][j])):
                poly_point = []
                poly_point_refs = []
                for l in range(len(poly_dict['features'][i]['geometry']['coordinates'][j][k])):
                    poly_point.append((list(reversed(poly_dict['features'][i]['geometry']['coordinates'][j][k][l]))))
                    grid_ref = latlong2grid(poly_point[l][1], poly_point[l][1])
                    poly_point_refs.append([grid_ref.E, grid_ref.N])
                full_poly.append(poly_point)
                full_grid_poly.append(poly_point_refs)
            j_list.append(j)
            poly_coords.append(full_poly)
            poly_grid_refs.append(full_grid_poly)
    for i in j_list:
        if i != 0:
            print(i)
    return poly_coords, poly_grid_refs

def make_grid(polygon, n_points, min_point, max_point):
    x_cen = polygon[0][0]
    y_cen = polygon[0][1]
    x = np.outer(np.linspace(min_point + x_cen, x_cen + max_point, n_points), np.ones(n_points))
    y = np.outer(np.linspace(min_point + y_cen, y_cen + max_point, n_points), np.ones(n_points)).T
    z = np.zeros(x.shape)
    return x, y, z

def add_polygon(polygon, x, y, z, args):
    polygon.append(polygon[0])
    points = np.concatenate((np.reshape(x, (x.size, 1)), np.reshape(y, (y.size, 1))), axis = 1)
    path = mpltPath.Path(polygon)
    poly_bool = np.reshape(np.array(path.contains_points(points)), x.shape)
    update_adjacent(z, poly_bool, args)
    return x, y, z

def update_adjacent(z, poly_bool, args):
    axis_list, shift_list, value_list = args
    for i in range(10):
        for axis in axis_list:
            z = shift_one_dim(z, poly_bool, axis, shift_list[i], value_list[i])
    return z

def shift_one_dim(z, poly_bool, axis, shift, value):
    shift_bool = np.roll(poly_bool, shift, axis)
    zero_bool = np.reshape([z==0], z.shape)
    shift_bool[:, [0, -1]] = False
    shift_bool[[0, -1], :] = False
    z[shift_bool *zero_bool] = value
    return z

poly_coords, poly_grid_refs = get_poly_coords(pub_poly_dict)
polygon = poly_grid_refs[0][0]
n_points = 1000
min_point = -5 
max_point = 5
axis_list = [0, 0, 1, 1, (1, 1), (-1, 1), (1, -1), (-1, -1)]
shift_list = np.linspace(1, 100, 10, dtype= np.int64)
value_list = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
args = [axis_list, shift_list, value_list]

x, y, z = make_grid(polygon, n_points, min_point, max_point)
for i in range(len(poly_grid_refs)):
    x, y, z = add_polygon(poly_grid_refs[i][0], x, y, z, args)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='inferno')
plt.show()
