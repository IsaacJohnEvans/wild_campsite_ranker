from flask import Flask, render_template, url_for, jsonify, request, json
import re
import math
app = Flask(__name__)
from basic_weather_calls import weather_mesh
from wind_shelter import wind_shelter
from OSGridConverter import latlong2grid

class Optimiser():
    def __init__(self, latlon, zoom_level, bbox, features, preferences):
        self.preferences = self.updatePreferences(preferences)
        self.latlon = latlon
        self.zoom_level = zoom_level
        self.bbox = self.getBBoxList(bbox)
        self.features = features
        self.shelterIndex = self.getShelterIndex()
        self.OSGridReference = self.getOSGridReference()
        self.tempWind = self.getTempWind()
        self.printStats()

    def getBBoxList(self, bbox):
        bboxLatLon = re.findall('\(.*?\)', bbox)
        bboxList = []
        for latLon in bboxLatLon:
            bboxList.append(latLon.replace('(','').replace(')','').replace(' ','').split(','))

        return bboxList

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

@app.route('/set_preferences', methods=['POST', 'GET'])
def get_preferences():
    if request.method == 'POST':
        preferences = request.form['preferences']
        data = {'status':"success"}
        try:
            optimiser.preferences = optimiser.updatePreferences(preferences)
            print(optimiser.preferences, flush=True)
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

        global optimiser
        optimiser = Optimiser(latlon, zoom_level, bbox, features, preferences)
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
        
    return data, 200 # 200 tells ajax "success!"
   
def get_num_features(feats):
    dictionary = json.loads(feats)
    num = len(dictionary)
    return num


if __name__ == "__main__":
    
    app.run(debug=True)