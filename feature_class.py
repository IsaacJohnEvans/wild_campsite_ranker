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
    def __init__(self, feature_id, feature_type, shape_type, feature_grid_refs):
        self.number = feature_id
        self.feature_type = feature_type
        self.shape_type = shape_type
        self.shape = feature_grid_refs
    
class map_layer:
    def __init__(self, x, y, z, name):
        self.features = []
        self.x = x
        self.y = y
        self.z = z
        self.layer_name = name
        self.poly_bool = np.zeros(z.shape).astype(bool)
    
    def update_poly_bool(self, poly_bool):
        self.poly_bool = np.logical_or(self.poly_bool, poly_bool)
        
    def bool_feature(self, feat):
        if feat.shape_type == 'Point':
            pass
        elif feat.shape_type == 'LineString':
            pass
        elif feat.shape_type == 'MultiLineString':
            pass
        elif feat.shape_type == 'Polygon':
            self.polygon_to_points(self.x, self.y, feat.shape)
        elif feat.shape_type == 'MultiPolygon':
            for poly in feat.shape:
                self.polygon_to_points(self.x, self.y, poly)

    def polygon_to_points(self, polygon):
        points = np.concatenate((np.reshape(self.x, (self.x.size, 1)), np.reshape(self.y, (self.y.size, 1))), axis = 1)
        path = mpltPath.Path(polygon)
        self.update_poly_bool(self, np.reshape(np.array(path.contains_points(points)), self.x.shape))
        
    def make_dilate_struct():
        struct = np.ones((3, 3))
        struct[1, 1] = 0
        return struct.astype(bool)

    def dilate_layer(self, layer1, struct, value):
        layer2 = ndimage.binary_dilation(layer1, structure=struct)
        self.grid[2][np.logical_and(layer2, np.logical_not(layer1.astype(bool)))] = value
        return layer2, self.grid[2]
    
    def dilate_poly(self, poly_bool, z, struct, values, sigma):
        layer2, self.grid[2] = self.dilate_layer(self.grid[2], poly_bool, struct, values[0])
        for val in values[1:]:
            layer2, self.grid[2] = self.dilate_layer(self.grid[2], layer2, struct, val)
        self.grid[2] = skimage.filters.gaussian(self.grid[2], sigma)


class map_class:
    def __init__(self):
        self.grid = []
    def make_grid(self, x_cen, y_cen, n_points, min_point, max_point):
        x = np.outer(np.linspace(min_point + x_cen, x_cen + max_point, n_points), np.ones(n_points))
        y = np.outer(np.linspace(min_point + y_cen, y_cen + max_point, n_points), np.ones(n_points)).T
        z = np.zeros(x.shape)
        self.grid = [x, y, z]