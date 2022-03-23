#coding:utf8
#%%

import folium
from folium import plugins
from matplotlib.font_manager import json_load
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import overpy
import json
from tqdm import tqdm
from math import e, pi
from OSGridConverter import latlong2grid

#%%
def get_query_response(query_str):
    """
    :param query_str: String representing Overpass query
    :return: JSON response
    """
    api = overpy.Overpass()
    response = api.query(query_str)
    return response

def get_data(query):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = query
    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    return response

def get_coords(data):
    # Collect coords into list
    coords = []
    for element in data['elements']:
        if element['type'] == 'node':
            lon = element['lon']
            lat = element['lat']
            coords.append((lat, lon))
        elif 'center' in element:
            lon = element['center']['lon']
            lat = element['center']['lat']
            coords.append((lat, lon))
    # Convert coordinates into numpy array
    X = np.array(coords, dtype=np.float64)
    return X

def uk_pubs():
    query = \
        """             [out:json];
                        (area["ISO3166-1"="GB"][admin_level=2];)->.a;
                        (node["amenity"="pub"](area);
                         way["amenity"="pub"](area);
                         rel["amenity"="pub"](area);
                        );
                        out center;"""
    

    response = get_query_response(query)
    return response

def process_nodes(data):
    nrows = len(data.nodes)
    X = np.zeros((nrows, 3), dtype=np.float32)
    for i in range(nrows):
        X[i, 1] = data.nodes[i].lat
        X[i, 2] = data.nodes[i].lon
        X[i, 0] = data.nodes[i].id
    return X

def nearest_feat(loc, X):
    dist = ((X[:, 1] - loc[0])**2 + (X[:, 2] - loc[1])**2) **0.5
    return X[np.argmin(dist), :]

def conc_feat():
    return

def plot_heatmap(X, centre_coords = [ 51.45, -2.65], zoom_start = 7, radius = 10):
    m = folium.Map(centre_coords, zoom_start=zoom_start)
    m.add_child(plugins.HeatMap(X, radius=radius))
    return m

#%%
api = overpy.Overpass()
Bristol_feat = api.query("[out:json];nwr(51.44,-2.7,51.46,-2.6);out;")
Bristol_X = process_nodes(Bristol_feat)
Bristol_feat.nodes
plot_heatmap(Bristol_X, zoom_start= 12)

#%%

UK_pubs = uk_pubs()
pub_X = process_nodes(UK_pubs)
loc = np.array([51.44927, -2.61905])
print(nearest_feat(loc, pub_X))
print((pub_X[:, 0] == 279444017).any())

#%%
query = \
        """             [out:json];
                        (area["ISO3166-1"="GB"][admin_level=2];)->.a;
                        (node["amenity"="pub"](area);
                         way["amenity"="pub"](area);
                         rel["amenity"="pub"](area);
                        );
                        out center;"""
pub_dict = get_data(query)
#%%
pub_X = process_nodes(pub_dict)
plot_heatmap(pub_X)
pub_X.shape

#%%
query = \
    """nwr(51.44,-2.7,51.46,-2.6);
    [out:json];"""
api = overpy.Overpass()
Bristol_feat = api.query(query)
#%%
api = overpy.Overpass()
result = api.query("node(51.44,-2.7,51.46,-2.6);out;")
#%%
data = result
nrows = len(data.nodes)
X = np.zeros((nrows, 3), dtype=np.float32)
for i in tqdm(range(nrows)):
    X[i, 1] = data.nodes[i].lat
    X[i, 2] = data.nodes[i].lon
    X[i, 0] = data.nodes[i].id
#%%
result.nodes[0].id
X[i, :] = data.nodes[i].lat
X[i, :] = data.nodes[i].lon
X[i, :] = data.nodes[i].id

#%%
import overpass
api = overpass.API()

#%%
var = 0.1
mean = 1
n_points = 100
x = np.outer(np.linspace(-2, 2, n_points), np.ones(n_points))
y = x.copy().T
x_cen = 0
y_cen = 0
z = np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)
z = np.tanh(z)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
plt.show()

#%%
x = 0
y = 0 
x_cen = 0
y_cen = 0
mean = 0

np.exp(-0.5*((((x-x_cen)**2 + (y-y_cen)**2)**0.5 - mean)/var)**2)/(var*(2 * pi)**0.5)
np.tanh(0)
#%%
import json
with open('bristol_test_data/bristol_pubs_points.geojson', "r") as read_file:
    pub_dict = json.load(read_file)

def get_point_coords(data_dict):
    coords = []
    grid_refs = []
    for i in range(len(data_dict['features'])):
        coords.append(list(reversed(data_dict['features'][i]['geometry']['coordinates'])))
        grid_ref = latlong2grid(coords[i][0], coords[i][1])
        grid_refs.append([grid_ref.E, grid_ref.N])
    coords = np.array(coords)
    grid_refs = np.array(grid_refs)
    return coords, grid_refs

pub_coords, pub_grid_refs = get_point_coords(pub_dict)
pub_grid_refs
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
                    grid_ref = latlong2grid(poly_point[0][0], poly_point[0][1])
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

poly_coords, poly_grid_refs = get_poly_coords(pub_poly_dict)
print(len(poly_coords), len(poly_coords[0]), len(poly_coords[0][0]), len(poly_coords[0][0][0]), type(poly_coords[0][0][0][0]))
