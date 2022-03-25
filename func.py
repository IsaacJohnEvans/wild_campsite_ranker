#coding utf8
#%%
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from math import e, pi
import matplotlib.path as mpltPath
import json
from OSGridConverter import latlong2grid
from scipy.spatial.distance import cdist, pdist, directed_hausdorff
from scipy import ndimage
import skimage

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
    return poly_coords, poly_grid_refs

def make_grid(x_cen, y_cen, n_points, min_point, max_point):
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
    for i in range(len(shift_list)):
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
#%%
poly_coords, poly_grid_refs = get_poly_coords(pub_poly_dict)
#%%

polygon = poly_grid_refs[0][0]
polygon = [[0, 0,],
           [0, 1],
           [1, 1],
           [1, 0],
           [0, 0]]

n_points = 1000
min_point = -5
max_point = 5
x_cen = polygon[0][0]
y_cen = polygon[0][1]
axis_list = [0, 0, 1, 1, (1, 1), (-1, 1), (1, -1), (-1, -1)]
shift_list = np.linspace(-10, 10, 20, dtype= np.int64)
value_list = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0] *2
args = [axis_list, shift_list, value_list]

x, y, z = make_grid(x_cen, y_cen, n_points, min_point, max_point)

x, y, z = add_polygon(polygon, x, y, z, args)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='inferno')
plt.show()

#%%
'''
for i in range(len(poly_grid_refs)):
    x, y, z = add_polygon(poly_grid_refs[i][0], x, y, z, args)
'''
#%%
a = np.zeros((10, 10))
a[5, 5] = 1

a.astype(bool)
axis_list = [0, 1]
[0, 1, 1, ]
shift_list = np.linspace(-5, 5, 10, dtype= np.int64)
value_list = [0, 1, 2, 1, 0] * 2
args = [axis_list, shift_list, value_list]
update_adjacent(a, a.astype(bool), args)
a

#%%


def dilate_layer(layer1, z, struct, value):
    layer2 = ndimage.binary_dilation(layer1, structure=struct)
    z[np.logical_and(layer2, np.logical_not(layer1.astype(bool)))] = value
    return layer2, z



n_points = 40
min_point = -400
max_point = 400
x_cen = polygon[0][0]
y_cen = polygon[0][1]
a = np.zeros((n_points, n_points))
z = np.zeros((n_points, n_points))
x, y, z = make_grid(x_cen, y_cen, n_points, min_point, max_point)
a[17:20, 7] = 1
a[17, 18:23] = 1
a[14:17, 22] = 1
a[11:13, 11:13] = 1
a = np.zeros((n_points, n_points))
a[15:25, 15:25] = 1
struct2 = ndimage.generate_binary_structure(2, 2)
struct = np.ones((3, 3))
struct[1, 1] = 0
struct = struct.astype(bool)

norm = np.linspace(0, 2, 11)
mean = 1
var = 1
norm = np.exp(-0.5*((norm - mean)/var)**2)/(var*(2 * pi)**0.5)

value_list = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
dist = 200
value_list = -np.cos(np.linspace(0, 2*np.pi, int(dist* n_points/(max_point-min_point)))) +1
layer2, z = dilate_layer(a, z, struct, value_list[0])

for val in value_list[1:]:
    layer2, z = dilate_layer(layer2, z, struct, val)

z = skimage.filters.gaussian(z, 5)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='inferno')
plt.show()

#%%
polygon = poly_grid_refs[0][0]
polygon
#%%
n_points = 20
min_point = -10
max_point = 10
x_cen = polygon[0][0] + 5
y_cen = polygon[0][1] + 5

x, y, z = make_grid(x_cen, y_cen, n_points, min_point, max_point)
points = np.concatenate((np.reshape(x, (x.size, 1)), np.reshape(y, (y.size, 1))), axis = 1)
path = mpltPath.Path(polygon)
poly_bool = np.reshape(np.array(path.contains_points(points)), x.shape)
print(poly_bool)
#%%
layer2, z = dilate_layer(poly_bool, z, struct, 1)

for val in [2, 3, 2, 1, 0]:
    layer2, z = dilate_layer(layer2, z, struct, val)

z = skimage.filters.gaussian(z, 1)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='inferno')
plt.show()
z