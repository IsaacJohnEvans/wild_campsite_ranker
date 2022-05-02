# coding : utf8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import json
from OSGridConverter import latlong2grid
from scipy import ndimage
import skimage
from shapely import wkt
from tqdm import tqdm
import pandas as pd

class map_feature:
    """
    A class to represent a map feature.

    Variables:
    Feature ID: The ID of the feature
    Feature Type: The type of the feature
    Shape Type: The type of the shape out of the following:
        Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
    latlong: A list of tuples of the latitudes and longitudes of the feature
    Shape: A list of tuples of the grid references of the feature
    """

    def __init__(self, feature_id, feature_type, shape_type, latlong):
        self.number = feature_id
        self.feature_type = feature_type
        self.shape_type = shape_type
        self.latlong = latlong
        self.shape = []
        if self.shape_type == "Point":
            grid_ref = latlong2grid(self.latlong[1], self.latlong[0])
            self.shape.append([grid_ref.E, grid_ref.N])
        elif self.shape_type in ["MultiPoint", "LineString"]:
            for i in self.latlong:
                grid_ref = latlong2grid(i[1], i[0])
                self.shape.append([grid_ref.E, grid_ref.N])
        elif self.shape_type in ["MultiLineString", "Polygon"]:
            for i in self.latlong:
                grid_refs = []
                for j in i:
                    grid_ref = latlong2grid(j[1], j[0])
                    grid_refs.append([grid_ref.E, grid_ref.N])
                self.shape.append(grid_refs)
        elif self.shape_type == "MultiPolygon":
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
    """
    A class of a map layer with a list of features to be added to the map.

    Variables:

    """

    def __init__(self, grid, name, effect, distance, features):
        self.features = features
        self.grid = grid
        self.points = np.concatenate(
            (
                np.reshape(self.grid[0], (self.grid[0].size, 1)),
                np.reshape(self.grid[1], (self.grid[1].size, 1)),
            ),
            axis=1,
        )
        self.layer_name = name
        self.sigma = 1
        self.effect = effect
        self.dist = distance
        self.values = self.effect_values()
        self.poly_bool = np.zeros(self.grid[2].shape).astype(bool)

    def effect_values(self):
        x = np.linspace(0, 2 * np.pi, self.dist)
        y = self.effect / 2 * (-np.cos(x) + 1)
        return y

    def bool_features(self):
        for feat in self.features:
            if feat.shape_type == 'Point':
                poly_point = np.logical_and(self.grid[0] == feat.shape[0][0], self.grid[1] == feat.shape[0][1])
                self.poly_bool = np.logical_or(self.poly_bool, poly_point)
            
            elif feat.shape_type == 'LineString':
                self.polygon_to_points(feat.shape)
            elif feat.shape_type == 'MultiLineString':
                for i in feat.shape:
                    self.polygon_to_points(i)
            elif feat.shape_type == 'Polygon':
                self.polygon_to_points(feat.shape[0])
            else:
                if feat.shape_type == "MultiPolygon":
                    for poly in feat.shape:
                        self.polygon_to_points(poly[0])
                        self.polygon_to_points(poly[0])
             
            '''
            Also because this is a for loop it will only not add to the last layer 
            so need a features there absolutely can't camp there numpy boolean array
            '''
    def polygon_to_points(self, polygon):
        path = mpltPath.Path(polygon)
        new_poly_bool = np.reshape(
            np.array(path.contains_points(self.points)), self.poly_bool.shape
        )
        self.poly_bool = np.logical_or(self.poly_bool, new_poly_bool)
    
    def dilate_layer(self, layer1, struct, value):
        layer2 = ndimage.binary_dilation(layer1, structure=struct)
        self.grid[2][
            np.logical_and(layer2, np.logical_not(layer1.astype(bool)))
        ] = value
        return layer2

    def dilate_poly(self, struct):
        layer2 = self.dilate_layer(self.poly_bool, struct, self.values[0])
        for val in self.values[1:]:
            layer2 = self.dilate_layer(layer2, struct, val)
        self.grid[2] = skimage.filters.gaussian(self.grid[2], self.sigma)
  
class heatmap_layer():
    def __init__(self, bbox, preferences = []):
        pd.DataFrame(np.array(bbox)).to_csv('bbox.csv', index = False, header = False)
        NW_gr = latlong2grid(bbox[0][1],bbox[1][0])
        NW = np.array([NW_gr.E, NW_gr.N])
        SE_gr = latlong2grid(bbox[1][1],bbox[0][0])
        SE = np.array([SE_gr.E, SE_gr.N])
        n_points = NW - SE
        NW[1] = SE[1] + n_points[0]
        n_points = NW - SE
        print(n_points)
        x = np.outer(np.linspace(SE[0], NW[0], 1 + n_points[0]), np.ones(1 + n_points[0]))
        y = np.outer(np.linspace(SE[1], NW[1], 1 + n_points[0]), np.ones(1 + n_points[0])).T
        z = np.zeros(x.shape)
        self.grid = [x, y, z]
        self.get_features()
        self.get_unique_feature_types()
        self.layers = []
        self.get_preference_features(preferences)

    def get_preference_features(self, preferences):
        self.preference_features = preferences
        '''
        This needs to be a dictionary with the feature type as the key and the preference as the value
        '''
    
    def get_features(self, file_name = 'data.geojson'):
        features = []
        with open(file_name, "r") as read_file:
            data_list = json.load(read_file)
        for i in range(len(data_list)):
            if "ele" in data_list[i]["properties"]:
                features.append(
                    map_feature(
                        i,
                        data_list[i]["properties"]["ele"],
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
            elif "FID" in data_list[i]["properties"]:
                features.append(
                    map_feature(
                        i,
                        data_list[i]["properties"]["FID"],
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
            elif "class" in data_list[i]["properties"]:
                features.append(
                    map_feature(
                        i,
                        data_list[i]["properties"]["class"],
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
            elif "water" in data_list[i]["layer"]["id"]:
                features.append(
                    map_feature(
                        i,
                        "water",
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
        self.features = features

    def get_unique_feature_types(self):
        desired_features = set()
        for feature in self.features:
            desired_features.add(feature.feature_type)
        self.unique_features = list(desired_features)
    
    def make_dilate_struct(self):
        struct = np.ones((3, 3))
        struct[1, 1] = 0
        return struct.astype(bool)

    def make_layers(self):
        effect = 1
        distance = 10
        grid = self.grid[:2]
        grid.append(np.zeros(self.grid[0].shape))
        layers = {}
        struct = self.make_dilate_struct()

        for unique_feature in self.unique_features:
            layers[unique_feature] = []
        
        '''
        Need to add a function to select features and set the importance of them using the slider data
        '''
        
        for feature in self.features:
            layers[feature.feature_type].append(feature)

        for unique_feature in tqdm(self.unique_features):
            layer1 = map_layer(
                grid, unique_feature, effect, distance, layers[unique_feature]
            )
            layer1.bool_features()
            layer1.dilate_poly(struct)
            self.grid[2] += layer1.grid[2]
            self.layers.append(layer1)
        
    def plot_heatmap(self):
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.grid[0], self.grid[1], self.grid[2], cmap="inferno")
        plt.show()


def main():    
    bbox = pd.read_csv('bbox.csv', header = None).to_numpy()
    heatmap = heatmap_layer(bbox)
    heatmap.make_layers()
    heatmap.plot_heatmap()
    
if __name__ == '__main__':
    main()
