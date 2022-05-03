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
    def __init__(self, bbox, preferences = None, n_points = None):
        self.unpack_bbox(bbox, n_points)
        self.make_grid()
        self.uncampable = self.grid[2].copy().astype(bool)
        self.layers = []
        self.preferences = preferences
        self.elevation = self.grid[2].copy()
        self.gradient = self.grid[2].copy()
    
    def unpack_bbox(self, bbox, n_points):
        '''
        A function to unpack the bounding box and turn it into OS grid references and the number of points in the grid.
        
        Parameters:
        bbox: The bounding box of the heatmap
        
        Returns:
        SE: The bottom left element
        NW: The top right element
        n_points: The number of points in the grid
        '''
        
        pd.DataFrame(np.array(bbox)).to_csv('bbox.csv', index = False, header = False)
        NW_gr = latlong2grid(bbox[0][1],bbox[1][0])
        NW = np.array([NW_gr.E, NW_gr.N])
        SE_gr = latlong2grid(bbox[1][1],bbox[0][0])
        SE = np.array([SE_gr.E, SE_gr.N])
        self.SE = SE
        self.NW = NW
        if n_points != None:
            self.n_points = (n_points, n_points)
        else:
            n_points = NW - SE
            NW[1] = SE[1] + n_points[0]
            n_points = NW - SE
            
            self.n_points = n_points
        
        
    def make_grid(self):
        '''
        A function to create the grid from the bounding box and the number of points in the grid.
        
        Returns:
        grid: A list of the data for the grid
        '''
        x = np.outer(np.linspace(self.SE[0], self.NW[0], 1 + self.n_points[0]), np.ones(1 + self.n_points[0]))
        y = np.outer(np.linspace(self.SE[1], self.NW[1], 1 + self.n_points[0]), np.ones(1 + self.n_points[0])).T
        z = np.zeros(x.shape)
        self.grid = [x, y, z]
    
    def get_features(self, file_name = 'data.geojson'):
        '''
        Extracts the features from the geojson file and creates map feature instances for each feature.
        
        Parameters:
        file_name: The name of the geojson file
        
        Returns:
        A list of the features on the heatmap
        '''
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
            elif "Water" in data_list[i]["layer"]["id"]:
                features.append(
                    map_feature(
                        i,
                        "Water",
                        data_list[i]["geometry"]["type"],
                        data_list[i]["geometry"]["coordinates"],
                    )
                )
        self.features = features            

    def get_unique_feature_types(self):
        '''
        A function to find the unique feature types in the heatmap.
        
        Parameters:
        self.features: The features of the heatmap
        
        Returns:
        self.unique_features: The unique features of the heatmap
        '''
        desired_features = set()
        for feature in self.features:
            desired_features.add(feature.feature_type)
        self.unique_features = list(desired_features)
    
    def make_dilate_struct(self):
        '''
        Makes a structuring element for the dilation of the heatmap.
        '''
        struct = np.ones((3, 3))
        struct[1, 1] = 0
        return struct.astype(bool)

    def features_into_layers(self):
        '''
        A function to create a dictionary of unique layers which contain a list of the features of the heatmap.
        '''
        layers = {}
        for unique_feature in self.unique_features:
            layers[unique_feature] = []
        for feature in self.features:
            layers[feature.feature_type].append(feature)
        
        return layers
    
    def sort_features(self):
        '''
        A function to sort the features into good features and contours.
        '''
        good_features = []
        contour_lines = []
        
        for unique_feature in self.unique_features:
            if type(unique_feature) == str:
                good_features += [unique_feature]
            elif type(unique_feature) == int:
                contour_lines.append(unique_feature)
        
        self.unique_features = good_features
        self.contour_lines = sorted(contour_lines)
        
    def get_elevation(self, grid, effect, distance, layers):
        '''
        A function to calculate the elevation of the bbox area using the contours of the heatmap.
        Then a filter is applied to smooth between the layers.
        '''
        for contour in tqdm(self.contour_lines):
            distance = 1
            layer1 = map_layer(grid, contour, effect, distance, layers[contour])
            layer1.bool_features()
            self.elevation[layer1.poly_bool] = contour + 5
        self.elevation[self.elevation == 0] = self.contour_lines[0]
        self.elevation = skimage.filters.gaussian(self.elevation, sigma = 20)
    
    def get_gradient(self):
        '''
        A function to calculate the gradient of the bbox area.
        '''
        self.gradient = np.absolute(getSlopeMatrix(self.elevation))
        self.gradient = -(self.gradient / np.max(self.gradient))
    
    def no_preferences(self):
        '''
        A function to create a default dictionary of preferences for the heatmap when None is passed.
        The dictionary allocates a value of one to every unique feature type.
        '''
        self.preferences = {}
        for unique_feature in self.unique_features:
            self.preferences[unique_feature] = 1
    
    def unpack_preferences(self):
        '''
        A function to unpack the preferences from the website into a dictionary.
        The function changes the values of the dictionary to integers.
        Then each keyword is unpacked in turn.
        '''
        self.preferences = dict((k, int(v)-1) for k, v in self.preferences.items())

        if 'Shops' in self.preferences.keys():
            self.preferences = self.preferences | {'commercial_area': self.preferences['Shops'],
                                                   'food_and_drink_stores': self.preferences['Shops']}
        if 'Water' in self.preferences.keys():
            self.preferences = self.preferences | {'water': self.preferences['Water'], 
                                                   'stream' : self.preferences['Water'], 
                                                   'river': self.preferences['Water']}
        if 'Landmarks' in self.preferences.keys():
            self.preferences = self.preferences | {'landmark': self.preferences['Landmarks'], 
                                                   'historical': self.preferences['Landmarks'], 
                                                   'Place-like': self.preferences['Landmarks']}
        if 'Pubs' in self.preferences.keys():
            self.preferences = self.preferences | {'Pubs': self.preferences['Pubs'],
                                                   'food_and_drink' : self.preferences['Pubs']}
        if 'Paths' in self.preferences.keys():
            self.preferences = self.preferences | {'path': self.preferences['Paths'], 'track': self.preferences['Paths']}        
        
        if 'Accomodation' in self.preferences.keys():
            self.preferences = self.preferences | {'lodging': self.preferences['Accomodation']}
            
        if 'Medical' in self.preferences.keys():
            self.preferences = self.preferences | {'medical': self.preferences['Medical']}
        
        if set(self.preferences.keys()).intersection(self.unique_features) == set():
            print('No preferential features in the area selected. The heatmap will be based entirely on the contours of the area.')
    
    def sort_preferences(self, effect):
        '''
        A function that selects whether to generate preferences if there are none or to unpack the preferences.
        The function also prints the unique features and the preferences.
        '''
        if self.preferences == None:
            self.no_preferences()
        else:
            self.unpack_preferences()
        if 'elevation' not in self.preferences.keys():
            self.preferences = self.preferences | {'elevation': effect}
        print('Unique features: ', self.unique_features)
        print('Preferences: ', self.preferences)
        
    def get_bad_features(self, grid, distance, layers):
        '''
        A function to get some features that are not good to camp on and turn them into points on a boolean array.
        '''
        self.bad_features = []
        self.bad_features += ['major_rail', 'minor_rail', 'primary', 'secondary', 'tertiary', 'wetland',
                              'arts_and_entertainment', 'residential', 'street', 'school', 'service']
        for unique_feature in set(self.bad_features).intersection(set(layers.keys())):
            layer1 = map_layer(grid, unique_feature, 1, 
                distance, layers[unique_feature], 1)
            layer1.bool_features()
            self.uncampable = np.logical_or(self.uncampable, layer1.uncampable)
            self.uncampable[layer1.poly_bool] = 1
    
    def get_good_features(self, grid, distance, layers):
        '''
        A function to get the good features on the map and dilate an area around them.
        The function makes a dilating structure.
        For every layer the function creates a map_layer object.
            Next the function gets the features for a layer and turns them into points on a boolean array.
            Then the layer is dilated and the points are turned into an array.
            Then the values of the layer are added to the heatmap grid.        
        '''
        struct = self.make_dilate_struct()
        for unique_feature in tqdm(set(self.preferences.keys()).intersection(set(layers.keys()))):
            layer1 = map_layer(grid, unique_feature, 
                self.preferences[unique_feature], 
                distance, layers[unique_feature], 1)
            layer1.bool_features()
            self.uncampable = np.logical_or(self.uncampable, layer1.uncampable)
            self.uncampable[layer1.poly_bool] = 1
            layer1.dilate_poly(struct)
            self.grid[2] += layer1.grid[2]
            self.layers.append(layer1)
    
    def make_layers(self, distance = 100):
        '''
        A function to create a heatmap grid.
        
        Parameters:
        effect (int): The default effect of the data.
        distance (int): The distance of the dilation. (Half the distance provides the optimal distance.)
        
        The function creates a heatmap grid.
        Next the function gets the features and unique features.
        Then the features are sorted into layers.
        Next the elevation of the bbox is calculated and from that the gradient.
        The preferences for each feature are then defined.
        Then the good and bad features are turned into points on a boolean array.
        The good features are added to the heatmap and the bad features, including the gradient, are subtracted from the heatmap.
        '''
        
        effect = 1
        grid = self.grid[:2]
        grid.append(np.zeros(self.grid[0].shape))
        
        self.get_features()
        self.get_unique_feature_types()
        layers = self.features_into_layers()
        self.sort_features()
        self.get_elevation(grid, effect, distance =1, layers = layers)
        self.get_gradient()
        self.sort_preferences(effect)
        
        self.get_bad_features(grid, distance, layers)
        self.get_good_features(grid, distance, layers)
        
        self.grid[2] += self.gradient * self.preferences['elevation']
        self.grid[2][self.uncampable] -= 5
        
    def plot_3D_heatmap(self):
        '''
        Plots a 3D heatmap.
        '''
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.grid[0], self.grid[1], self.grid[2], cmap='inferno')
        plt.show()
    
    def plot_2D_heatmap(self):
        '''
        Plots a 2D heatmap.
        '''
        fig, ax = plt.subplots()
        heatmap_plot = ax.pcolor(self.grid[0], self.grid[1], self.grid[2], cmap='inferno', shading='auto')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(rotation = 45)
        plt.colorbar(heatmap_plot, label='Suitability for camping')
        plt.savefig('poster/optimiser_heatmap.pdf')
        plt.show()
        

def main():
    '''
    Runs an example heatmap using data.geojson and bbox.csv
    '''
    preferences = {'water': 2, 'elevation': 2}
    bbox = pd.read_csv('bbox.csv', header = None).to_numpy()
    heatmap = heatmap_layer(bbox)
    heatmap.make_layers(100)
    heatmap.plot_3D_heatmap()
    heatmap.plot_2D_heatmap()
    
if __name__ == '__main__':
    main()
