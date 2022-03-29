#coding utf8
#%%
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpltPath
import json
from OSGridConverter import latlong2grid
from scipy import ndimage
import skimage
from shapely import wkt

#%%
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
                    grid_ref = latlong2grid(poly_point[l][0], poly_point[l][1])
                    poly_point_refs.append(np.array([grid_ref.E, grid_ref.N]))
                full_poly.append(poly_point)
                full_grid_poly.append(poly_point_refs)
            j_list.append(j)
            poly_coords.append(full_poly)
            poly_grid_refs.append(full_grid_poly)
    return poly_coords, poly_grid_refs

def get_poly_centre(polygon):
    poly_str = 'POLYGON(('
    for i in tuple(map(tuple, polygon)):
        poly_str += str(i[0]) + ' ' + str(i[1]) + ', '
    poly_str = poly_str[:-2] + '))'
    p1 = wkt.loads(poly_str)
    x_cen = p1.centroid.coords[0][0]
    y_cen = p1.centroid.coords[0][1]
    return x_cen, y_cen

def get_min_max(polygon, x_cen, y_cen, values):
    min_point = np.round(np.min(polygon - np.array([x_cen, y_cen])), 0)*len(values)
    max_point = np.round(np.max(polygon - np.array([x_cen, y_cen])), 0)*len(values)
    return min_point, max_point

def make_grid(x_cen, y_cen, n_points, min_point, max_point):
    x = np.outer(np.linspace(min_point + x_cen, x_cen + max_point, n_points), np.ones(n_points))
    y = np.outer(np.linspace(min_point + y_cen, y_cen + max_point, n_points), np.ones(n_points)).T
    z = np.zeros(x.shape)
    return x, y, z

def polygon_to_points(x, y, polygon):
    points = np.concatenate((np.reshape(x, (x.size, 1)), np.reshape(y, (y.size, 1))), axis = 1)
    path = mpltPath.Path(polygon)
    poly_bool = np.reshape(np.array(path.contains_points(points)), x.shape)
    return poly_bool

def make_dilate_struct():
    struct = np.ones((3, 3))
    struct[1, 1] = 0
    struct = struct.astype(bool)
    return struct

def dilate_layer(layer1, z, struct, value):
    layer2 = ndimage.binary_dilation(layer1, structure=struct)
    z[np.logical_and(layer2, np.logical_not(layer1.astype(bool)))] = value
    return layer2, z

def dilate_poly(poly_bool, z, struct, values, sigma):
    layer2, z = dilate_layer(poly_bool, z, struct, values[0])
    for val in values[1:]:
        layer2, z = dilate_layer(layer2, z, struct, val)
    z = skimage.filters.gaussian(z, sigma)
    return z

def plot_poly(x, y, z):
    fig = plt.figure(2)
    ax = plt.axes(projection ='3d')
    ax.plot_surface(x, y, z, cmap ='inferno')
    plt.show()
#%%
with open('bristol_test_data/bristol_pubs_multipolygons.geojson', "r") as read_file:
    pub_poly_dict = json.load(read_file)
poly_coords, poly_grid_refs = get_poly_coords(pub_poly_dict)

#%%
polygon = poly_grid_refs[0][0]
polygon = np.array(polygon)
print(polygon)
plt.figure(1)
plt.plot(polygon[:,0], polygon[:,1], '-o')

n_points = 100

values = [1, 2, 3, 2, 1, 0]
sigma = 1

struct = make_dilate_struct()
x_cen, y_cen = get_poly_centre(polygon)
min_point, max_point = get_min_max(polygon, x_cen, y_cen, values)
min_point = -1000
max_point = 1000
x, y, z = make_grid(x_cen, y_cen, n_points, min_point, max_point)
poly_bool = polygon_to_points(x, y, polygon)
for i in range(len(poly_grid_refs)):
    polygon = poly_grid_refs[i][0]
    poly_bool = np.logical_or(poly_bool,polygon_to_points(x, y, polygon))

z = dilate_poly(poly_bool, z, struct, values, sigma)
plot_poly(x, y, z)
