# coding : utf8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.path as mpltPath
import json
from OSGridConverter import grid2latlong, latlong2grid, OSGridReference
from scipy import ndimage
import skimage
from shapely import wkt
from tqdm import tqdm
from matplotlib import cm
from elevation import getSlopeMatrix

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
        '''
        Initialisation of the map feature and conversion of latlong to grid references.
        '''
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

class map_layer(map_feature):
    """
    A class of a map layer with a list of features to be added to the map.
    The features are instances of the map_feature class.

    Variables:
    features: A list of map_feature instances
    grid: The grid on which the layer is to be placed
    points: A list of tuples of the grid references of the layer
    layer_name: The name of the layer 
    sigma: The sigma of the layer (for the gaussian filter)
    effect: The effect of the layer
    dist: The distance that the effect of the feature spreads out
    values: The values of the layer sampled from a sinusoid
    poly_bool: A boolean array of the shape of the grid indicating which points are features in the layer
    """

    def __init__(self, grid, name, effect, distance, features, sigma = 1):
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
        self.sigma = sigma
        self.effect = effect
        self.dist = distance
        self.values = self.effect_values()
        self.poly_bool = np.zeros(self.grid[2].shape).astype(bool)

    def effect_values(self):
        '''
        Creates a sinusoid of the effect of the shape of the layer.
        '''
        x = np.linspace(0, 2 * np.pi, self.dist)
        y = self.effect / 2 * (-np.cos(x) + 1)
        return y

    def bool_features(self):
        '''
        Creates a boolean array of the shape of the grid indicating which points on the grid are features in the layer.
        Uncampable is returned as the same as the poly bool array in order to indicate features as uncampable.
        '''
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
        self.uncampable = self.poly_bool
            
    def polygon_to_points(self, polygon):
        '''
        Converts a polygon to a list of points on the grid.
        '''
        path = mpltPath.Path(polygon)
        new_poly_bool = np.reshape(
            np.array(path.contains_points(self.points)), self.poly_bool.shape
        )
        self.poly_bool = np.logical_or(self.poly_bool, new_poly_bool)
    
    def dilate_layer(self, layer1, struct, value):
        '''
        Dilates a layer by the structuring element and assigns it a value.
        Parameters:
        layer1: The layer to be dilated
        struct: The structuring element
        value: The value to be assigned to the dilated layer
        
        returns:
        layer2 : The dilated layer
        '''
        layer2 = ndimage.binary_dilation(layer1, structure=struct)
        self.grid[2][np.logical_and(layer2, np.logical_not(layer1.astype(bool)))] = value
        return layer2

    def dilate_poly(self, struct):
        '''
        Dilation of a polygon by the structuring element followed by a filter of the layer using a gaussian filter.
        '''
        layer2 = self.dilate_layer(self.poly_bool, struct, self.values[0])
        for val in self.values[1:]:
            layer2 = self.dilate_layer(layer2, struct, val)
        self.grid[2] = skimage.filters.gaussian(self.grid[2], self.sigma)
  
class heatmap_layer():
    '''
    The heatmap layer.
    
    Parameters:
    bbox: The bounding box of the heatmap
    uncampable: The uncampable points of the heatmap
    grid: The grid of the heatmap compromised of x, y, and z
    layers: The layers of the heatmap
    preferences: The preferences of the heatmap
    features: The features of the heatmap
    unique_features: The unique features of the heatmap
    '''
    def __init__(self, bbox, preferences = None):
        pd.DataFrame(np.array(bbox)).to_csv('bbox.csv', index = False, header = False)
        NW_gr = latlong2grid(bbox[0][1],bbox[1][0])
        NW = np.array([NW_gr.E, NW_gr.N])
        SE_gr = latlong2grid(bbox[1][1],bbox[0][0])
        SE = np.array([SE_gr.E, SE_gr.N])
        n_points = NW - SE
        NW[1] = SE[1] + n_points[0]
        n_points = NW - SE
        x = np.outer(np.linspace(SE[0], NW[0], 1 + n_points[0]), np.ones(1 + n_points[0]))
        y = np.outer(np.linspace(SE[1], NW[1], 1 + n_points[0]), np.ones(1 + n_points[0])).T
        z = np.zeros(x.shape)
        self.uncampable = z.copy().astype(bool)
        self.grid = [x, y, z]
        self.get_features()
        self.get_unique_feature_types()
        self.layers = []
        self.preferences = preferences
        self.elevation = z.copy()
        self.gradient = z.copy()
    
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
        '''
            elif "FID" in data_list[i]["properties"]:
                features.append(
                    map_feature(
                        i,
                        data_list[i]["properties"]["FID"],
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
        '''
            

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
        grid = self.grid[:2]
        grid.append(np.zeros(self.grid[0].shape))
        layers = {}
        struct = self.make_dilate_struct()

        for unique_feature in self.unique_features:
            layers[unique_feature] = []
         
        for feature in self.features:
            layers[feature.feature_type].append(feature)
        
        good_features = []
        contour_lines = []
        
        for unique_feature in self.unique_features:
            if type(unique_feature) == str:
                good_features += [unique_feature]
            elif type(unique_feature) == int:
                contour_lines.append(unique_feature)
        self.unique_features = good_features
        contour_lines = sorted(contour_lines)
        
        for contour in tqdm(contour_lines):
            distance = 1
            layer1 = map_layer(
                grid,contour, effect, distance, layers[contour]
            )
            layer1.bool_features()
            self.elevation[layer1.poly_bool] = contour + 5
        
        self.elevation[self.elevation == 0] = contour_lines[0]
        self.elevation = skimage.filters.gaussian(self.elevation, sigma = 20)

        self.gradient = np.absolute(getSlopeMatrix(self.elevation))
        self.gradient = -(self.gradient / np.max(self.gradient))
        
        
        if self.preferences == None:
            self.preferences = {}
            for unique_feature in self.unique_features:
                self.preferences[unique_feature] = 10
        
        if set(self.preferences.keys()).intersection(self.unique_features) == set():
            print('No preferential features in the area selected.')
        
        #self.preferences = {'path': 100}
        #self.preferences = {'Food and '}
        print('Unique features: ', self.unique_features)
        print('Preferences: ', self.preferences)
        effect = 1
        for unique_feature in tqdm(self.preferences.keys()):
            distance = self.preferences[unique_feature]
            layer1 = map_layer(
                grid, unique_feature, effect, distance, layers[unique_feature], 1
            )
            layer1.bool_features()
            self.uncampable = np.logical_or(self.uncampable, layer1.uncampable)
            self.uncampable[layer1.poly_bool] = 10
            layer1.dilate_poly(struct)
            self.grid[2] += layer1.grid[2]
            self.layers.append(layer1)
        self.grid[2] += self.gradient         
        
        zero = np.zeros(self.grid[2].shape)
        zero[np.nonzero(self.grid[2])] = 1
        #print('Features everywhere = ',(self.uncampable != 0).all(), ' \n Grid nonzero in some places = ', zero.astype(bool).all())
        if (self.uncampable != 0) == (np.nonzero(self.grid[2])):
            self.grid[2][self.uncampable] = 0
        else:
            print('The uncampable points are the same as the features')
        
    def plot_heatmap(self):
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.grid[0], self.grid[1], self.grid[2], cmap='inferno')
        plt.show()
        #ax.plot_surface(self.grid[0], self.grid[1], self.uncampable, cmap='viridis')
        #plt.show()
        


def main():    
    bbox = pd.read_csv('bbox.csv', header = None).to_numpy()
    heatmap = heatmap_layer(bbox)
    heatmap.make_layers()
    x = heatmap.grid[0]
    y = heatmap.grid[1]
    z = heatmap.grid[2]
    n_spots = 5
    grid_spots = np.concatenate(
        (np.array([x[np.unravel_index(np.argsort(z.flatten())[-n_spots:], z.shape)[0], 0]]).T,
         np.array([y[0, np.unravel_index(np.argsort(z.flatten())[-n_spots:], z.shape)[1]]]).T),
        1)
    
    latlong_spots = []
    #print(grid2latlong(str(OSGridReference(grid_spots[0][0], grid_spots[0][1]))))
    for i in range(grid_spots.shape[0]):
        latlong = grid2latlong(str(OSGridReference(grid_spots[i][0], grid_spots[i][1])))
        latlong_spots.append([latlong.longitude, latlong.latitude])
    heatmap.plot_heatmap()
    
if __name__ == '__main__':
    main()
