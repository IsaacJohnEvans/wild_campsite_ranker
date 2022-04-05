#coding : utf8
#%%
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpltPath
import json
from OSGridConverter import latlong2grid
from scipy import ndimage
import skimage
from shapely import wkt

class map_feature:
    def __init__(self, feature_id, feature_type, shape_type, latlong):
        self.number = feature_id
        self.feature_type = feature_type
        self.shape_type = shape_type
        self.latlong = latlong
        self.shape = []
        if self.shape_type == 'Point':
            grid_ref = latlong2grid(i[1], i[0])
            self.shape.append([grid_ref.E, grid_ref.N])
        elif self.shape_type in ['MultiPoint', 'LineString']:
            for i in self.latlong:
                grid_ref = latlong2grid(i[1], i[0])
                self.shape.append([grid_ref.E, grid_ref.N])
        elif self.shape_type in ['MultiLineString', 'Polygon']:
            for i in self.latlong:
                grid_refs = []
                for j in i:
                    grid_ref = latlong2grid(j[1], j[0])
                    grid_refs.append([grid_ref.E, grid_ref.N])
                self.shape.append(grid_refs)
        elif self.shape_type == 'MultiPolygon':
            for i in self.latlong:
                poly_grid_refs = []
                for j in i:
                    grid_refs = []
                    for k in j:
                        grid_ref = latlong2grid(k[1], k[0])
                        grid_refs.append([grid_ref.E, grid_ref.N])
                    poly_grid_refs.append(grid_refs)
                self.shape.append(poly_grid_refs)
                
    def poly_latlong_to_grid(self, coords):
        grid_refs = []
        for i in coords:
            grid_ref = latlong2grid(i[1], i[0])
            grid_refs.append([grid_ref.E, grid_ref.N])
        return grid_refs

class map_layer(map_feature):
    def __init__(self, x, y, z, name, effect, distance, values):
        self.features = []
        self.x = x
        self.y = y
        self.z = z
        self.layer_name = name
        self.sigma = 1
        self.effect = effect
        self.dist = distance
        self.values = np.repeat(values, self.dist) * effect
        self.poly_bool = np.zeros(z.shape).astype(bool)
        
    def get_features(self, file_name = 'data.geojson'):
        with open(file_name, "r") as read_file:
            data_list = json.load(read_file)
        for i in range(len(data_list)):
            if 'ele' in data_list[i]['properties']:
                self.features.append(map_feature(i, data_list[i]['properties']['ele'],
                                                data_list[i]['geometry']['type'], 
                                               data_list[i]['geometry']['coordinates']))
            elif 'FID' in data_list[i]['properties']:
                self.features.append(map_feature(i, data_list[i]['properties']['FID'],
                                                data_list[i]['geometry']['type'], 
                                                data_list[i]['geometry']['coordinates']))
            elif 'class' in data_list[i]['properties']:
                self.features.append(map_feature(i, data_list[i]['properties']['class'],
                                                data_list[i]['geometry']['type'], 
                                                data_list[i]['geometry']['coordinates']))
            elif 'water' in data_list[i]['layer']['id']:
                self.features.append(map_feature(i, 'water',
                                                data_list[i]['geometry']['type'], 
                                                data_list[i]['geometry']['coordinates']))
    
    def update_poly_bool(self, poly_bool):
        self.poly_bool = np.logical_or(self.poly_bool, poly_bool)
    
    def add_feature_to_layer(self):
        for i in self.features:
            self.bool_feature(self, self.features[1])
        self.values = np.repeat(self.values* self.effect, self.dist)
        
    def bool_features(self):
        for feat in self.features:
            if feat.shape_type == 'Point':
                pass
            elif feat.shape_type == 'LineString':
                pass
            elif feat.shape_type == 'MultiLineString':
                pass
            elif feat.shape_type == 'Polygon':
                print(feat.shape)
                self.polygon_to_points(feat.shape[0])
            '''
            elif feat.shape_type == 'MultiPolygon':
                for poly in feat.shape:
                    self.polygon_to_points(poly[0])
            '''

    def polygon_to_points(self, polygon):
        points = np.concatenate((np.reshape(self.x, (self.x.size, 1)), np.reshape(self.y, (self.y.size, 1))), axis = 1)
        path = mpltPath.Path(polygon)
        poly_bool = np.reshape(np.array(path.contains_points(points)), self.x.shape)
        self.update_poly_bool(poly_bool)
        
    def make_dilate_struct(self):
        struct = np.ones((3, 3))
        struct[1, 1] = 0
        return struct.astype(bool)

    def dilate_layer(self, layer1, struct, value):
        layer2 = ndimage.binary_dilation(layer1, structure=struct)
        self.z[np.logical_and(layer2, np.logical_not(layer1.astype(bool)))] = value
        return layer2
    
    def dilate_poly(self, struct):
        layer2 = self.dilate_layer(self.poly_bool, struct, self.values[0])
        for val in self.values[1:]:
            layer2 = self.dilate_layer(layer2, struct, val)
        #self.z = skimage.filters.gaussian(self.z, self.sigma)
    
    def draw_heatmap(self):
        struct = self.make_dilate_struct()
        self.dilate_poly(struct)

    def plot_heatmap(self):
        fig = plt.figure(1)
        ax = plt.axes(projection ='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap ='inferno')
        plt.show()

class heatmap_layer():
    def __init__(self):
        self.grid = []
    def make_grid(self, x_cen, y_cen, n_points, min_point, max_point):
        x = np.outer(np.linspace(min_point + x_cen, x_cen + max_point, n_points), np.ones(n_points))
        y = np.outer(np.linspace(min_point + y_cen, y_cen + max_point, n_points), np.ones(n_points)).T
        z = np.zeros(x.shape)
        self.grid = [x, y, z]