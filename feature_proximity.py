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
from mpl_toolkits.mplot3d import Axes3D
#from feature_class import map_feature
#%%
def get_test_poly_coords(poly_dict):
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
                full_grid_poly.append(list_to_array(poly_point_refs))
            j_list.append(j)
            poly_coords.append(full_poly)
            poly_grid_refs.append(full_grid_poly)
    return poly_coords, poly_grid_refs

def list_to_array(a):
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))], dtype = np.float32)
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b

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
    
def add_feature(x, y, z, polygon, values, dist, effect, struct, sigma):
    values = np.repeat(values* effect, dist)
    polygon = poly_grid_refs[0][0]
    poly_bool = np.zeros(z.shape).astype(bool)
    min_points, max_points = [], []
    for i in range(len(poly_grid_refs)):
        min_point, max_point = get_min_max(polygon, x_cen, y_cen, values)
        min_points.append(min_point)
        max_points.append(max_point)
        polygon = poly_grid_refs[i][0]
        poly_bool = np.logical_or(poly_bool,polygon_to_points(x, y, polygon))
    z += dilate_poly(poly_bool, z, struct, values, sigma)
    return x, y, z, min_points, max_points
#%%
with open('bristol_test_data/bristol_pubs_multipolygons.geojson', "r") as read_file:
    pub_poly_dict = json.load(read_file)
poly_coords, poly_grid_refs = get_test_poly_coords(pub_poly_dict)
#%%
def get_poly_coords(file_name):
    with open(file_name, "r") as read_file:
        data_list = json.load(read_file)
    coords = {}
    grid_refs = {}
    count = 0
    data_types = ['Point', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
    for i in range(len(data_list)):
        if 'ele' in data_list[i]['properties']:
            coords[data_list[i]['properties']['ele']] = {}
            coords[data_list[i]['properties']['ele']][data_list[i]['geometry']['type']] = data_list[i]['geometry']['coordinates']
        elif 'FID' in data_list[i]['properties']:
            coords[data_list[i]['properties']['FID']] = {}
            coords[data_list[i]['properties']['FID']][data_list[i]['geometry']['type']] = data_list[i]['geometry']['coordinates']
        elif 'class' in data_list[i]['properties']:
            if data_list[i]['properties']['class'] not in coords.keys():
                coords[data_list[i]['properties']['class']] = {}
                if data_list[i]['geometry']['type'] not in coords[data_list[i]['properties']['class']].keys():
                    coords[data_list[i]['properties']['class']][data_list[i]['geometry']['type']] = []
                coords[data_list[i]['properties']['class']][data_list[i]['geometry']['type']].append(data_list[i]['geometry']['coordinates'])
    return data_list, coords, grid_refs

data_list, coords, grid_refs = get_poly_coords('data.geojson')

count = 0
for i in coords.keys():
    for j in coords[i].keys():
        for k in coords[i][j]:
            count += 1
            print(i, j, k)
print(count, len(data_list))
#coords, grid_refs

#%%
poly_coords, poly_grid_refs = get_test_poly_coords(pub_poly_dict)


#%%
polygon = poly_grid_refs[0][0]
plt.figure(1)
plt.plot(polygon[:,0], polygon[:,1], '-o')

dist = 10
n_points = 500
effect = 2
values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0])

sigma = 10
struct = make_dilate_struct()

x_cen, y_cen = get_poly_centre(polygon)
min_point = -15000
max_point = 15000
x, y, z = make_grid(x_cen, y_cen, n_points, min_point, max_point)
x, y, z, min_points, max_points = add_feature(x, y, z, poly_grid_refs, values, dist, effect, struct, sigma)
plot_poly(x, y, z)

#%%
min(min_points), max(max_points)

#%%
print(z[z==np.max(z)].shape,np.unravel_index(np.argmax([z==np.max(z)], keepdims=True), z.shape))

#%%
file_name = 'data.geojson'
with open(file_name, "r") as read_file:
    data_list = json.load(read_file)
coords = {}
grid_refs = {}
count = 0
map_features = []
'''
data_types = ['Point', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
for i in range(len(data_list)):
    
    if 'ele' in data_list[i]['properties']:
        map_features.append(map_feature(i, data_list[i]['properties']['ele'],
                                        data_list[i]['geometry']['type'], 
                                        data_list[i]['geometry']['coordinates']))
    elif 'FID' in data_list[i]['properties']:
        map_features.append(map_feature(i, data_list[i]['properties']['FID'],
                                        data_list[i]['geometry']['type'], 
                                        data_list[i]['geometry']['coordinates']))
    elif 'class' in data_list[i]['properties']:
        map_features.append(map_feature(i, data_list[i]['properties']['class'],
                                        data_list[i]['geometry']['type'], 
                                        data_list[i]['geometry']['coordinates']))
    elif 'water' in data_list[i]['layer']['id']:
        map_features.append(map_feature(i, 'water',
                                        data_list[i]['geometry']['type'], 
                                        data_list[i]['geometry']['coordinates']))
    else:
        print(data_list[i])

for i in map_features:
    print(i.shape_type)
'''