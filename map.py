from flask import Flask, render_template, url_for, jsonify, request, json
import re
import math
app = Flask(__name__)
from basic_weather_calls import weather_mesh
from wind_shelter import wind_shelter
from OSGridConverter import latlong2grid
from pathfinding import get_tile
from feature_class import map_feature, map_layer, heatmap_layer
import mercantile
from elevation import getElevationMatrix, rasterToImage, getRasterRGB ,getSlopeMatrix
from pathfinding import construct_lng_lat_matrix, get_min_path
import numpy as np
class Optimiser():
    def __init__(self):
        self.preferences = None
        self.latlon = None
        self.zoom_level = None
        self.bbox = None
        self.startPoint = None
        self.endPoint = None
        self.features = None
        self.shelterIndex = None
        self.OSGridReference = None
        self.tempWind = None
        
    
    def updateOptimiser(self, latlon, zoom_level, bbox, features, preferences):
        self.latlon = latlon
        self.zoom_level = float(zoom_level)
        self.bbox = self.getBBoxList(bbox)
        self.features = features
        self.preferences = self.updatePreferences(preferences)
        self.shelterIndex = self.getShelterIndex()
        self.OSGridReference = self.getOSGridReference()
        self.tempWind = self.getTempWind()
        self.printStats()
        
        self.convertToJson(get_min_path(self.bbox[0], self.bbox[1], math.floor(self.zoom_level)))
        #print(self.minPathToPoint, flush=True)
    def make_heatmap(self):
        '''
        Need to set npoints based on the zoom level
        '''
        print('Making heatmap, please wait')
        n_points = 1000
        heatmap = heatmap_layer(self.bbox, n_points)
        heatmap.make_layers()
        heatmap.plot_heatmap()

    def convertToJson(self, minPath):
        geojson = {
            "type": "FeatureCollection",
            "features": [
            {
                "type": "Feature",
                "geometry" : {
                    "type": "LineString",
                    "coordinates": minPath
                    }}]}
        output = open("minPath.geojson", 'w')
        json.dump(geojson, output)
        return geojson
    def getBBoxList(self, bbox):
        bboxLatLon = re.findall('\(.*?\)', bbox)
        bboxList = []
        for latLon in bboxLatLon:
            bboxList.append(latLon.replace('(','').replace(')','').replace(' ','').split(','))
        bbox = [[float(bboxList[0][0]), float(bboxList[0][1])],[float(bboxList[1][0]),float(bboxList[1][1])]]
        return bbox
    
    def setPoint(self, latlonDict, pointType):
        if pointType == "start":
            self.startPoint = [latlonDict['lat'], latlonDict['lng']]
            
        else:
            self.endPoint = [latlonDict['lat'], latlonDict['lng']]
        if self.startPoint != None and self.endPoint != None:
            min_path = get_min_path(self.startPoint, self.endPoint, math.ceil(float(self.zoom_level)))
            return self.convertToJson(min_path)
        else:
            return "False"
    def getFeatures(self):
        pass
        #print(self.features)
    def getShelterIndex(self):
        shelterIndex = wind_shelter(self.latlon['lat'], self.latlon['lng'], math.ceil(float(self.zoom_level)))
        return shelterIndex
    def getOSGridReference(self):
        return str(latlong2grid(self.latlon['lat'], self.latlon['lng']))
    def getTempWind(self):
        get_weather = weather_mesh([self.latlon['lat']], [self.latlon['lng']])
        tempWind = get_weather['features'][0]['properties']
        return tempWind
    def updatePreferences(self, newPreferences):
        preferences = []
        for i in newPreferences:
            if i.isdigit():
                preferences.append(i)
        return preferences
    def printStats(self):
        print("Latlon:",self.latlon, flush=True)
        print("zoom_level:",self.zoom_level, flush=True)
        print("bbox:",self.bbox, flush=True)
        print("shelterIndex:",self.shelterIndex, flush=True)
        print("OSGridReference:",self.OSGridReference, flush=True)
        print("preferences:",self.preferences, flush=True)
@app.route('/')
def home():
    return render_template('bivouac.html')
@app.route('/start_destination', methods = ['POST','GET'])
def start_destination():
    if request.method == 'POST':
        location = request.form['location']
            
        minpath = optimiser.setPoint(json.loads(re.findall('\{.*?\}',location)[1]),"start")
        
        data = {'status':"success", "minpath":minpath}
    return data, 200
@app.route('/end_destination', methods = ['POST','GET'])
def end_destination():
    if request.method == 'POST':
        location = request.form['location']
        minpath = optimiser.setPoint(json.loads(re.findall('\{.*?\}',location)[1]), "end")
        data = {'status':"success",
        "minpath": minpath}
    return data, 200
@app.route('/set_preferences', methods=['POST', 'GET'])
def get_preferences():
    if request.method == 'POST':
        preferences = request.form['preferences']
        data = {'status':"success"}
        try:
            optimiser.preferences = optimiser.updatePreferences(preferences)
            
        except NameError:
            pass
        #print(preferences, flush=True)
    
    return data, 200
@app.route('/get_result', methods=['POST', 'GET'])
def process_result():
    if request.method == 'POST':
        
        mouse_pos = request.form['mouse_info']
        zoom_level = request.form['zoom_level']
        bbox = request.form['bbox']
        preferences = request.form['vals']
        features = request.form['features']
        # print(features)
        with open('data.geojson', 'w') as f:
            json.dump(json.loads(features), f)
        latlon = json.loads(re.findall('\{.*?\}',mouse_pos)[1])
        optimiser.updateOptimiser(latlon, zoom_level, bbox, json.loads(features), preferences)
        # optimiser.make_heatmap()
        # print("Output :" + mouse_pos, flush=True)
        # print("Zoom level :" + zoom_level, flush=True)
        # print("Features :" + features, flush=True)
        # Define loads of interesting things here, ie list of coords to plot that is a path
        num_features = get_num_features(features)
        # add whatever keys and values we want to this
        data = {"status": "success",
            "some": num_features,
            "temp": optimiser.tempWind['Temp'],
            "wind": optimiser.tempWind['Wind'],
            "wind_shelter": optimiser.shelterIndex,
            "osGrid": optimiser.OSGridReference
            }
        # creating elevation matrix (needs to be using the bbox and latlon centre)
        MAPBOX_TOKEN = 'pk.eyJ1IjoiY3Jpc3BpYW5tIiwiYSI6ImNsMG1oazJhejE0YzAzZHVvd2Z1Zjlhb2YifQ.cv0zlPYY6WnoKM9YLD1lMQ'
        #tile_coords = mercantile.tile(bbox[0][0], bbox[0][1], zoom_level)
        '''tile_coords = mercantile.tile(bbox[0][0], bbox[0][1], zoom_level)
        upper_left = mercantile.ul(tile_coords)
        lnglat_mat = construct_lng_lat_matrix(upper_left, zoom_level)
        elevation_mat = getElevationMatrix(MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y)
        slope_mat = getSlopeMatrix(elevation_mat)'''
        
        
    return data, 200 # 200 tells ajax "success!"
   
def get_num_features(feats):
    dictionary = json.loads(feats)
    num = len(dictionary)
    return num

if __name__ == "__main__":
    global optimiser
    optimiser = Optimiser()
    app.run(debug=True)