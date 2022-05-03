from tkinter import OptionMenu
from flask import Flask, render_template, url_for, jsonify, request, json
import re
import math

app = Flask(__name__)
from basic_weather_calls import weather_mesh
from wind_shelter import wind_shelter
from OSGridConverter import *

# from pathfinding import get_tile
from feature_class import map_feature, map_layer, heatmap_layer
import mercantile

# from elevation import getElevationMatrix, rasterToImage, getRasterRGB, getSlopeMatrix
from new_pathfinding import new_get_min_path
import numpy as np


class Optimiser:
    def __init__(self):
        self.preferences = {
            "Test1": None,
            "Test2": None,
            "Test3": None,
            "Test4": None,
            "Test5": None,
            "Test6": None,
            "Test7": None,
            "Test8": None,
        }
        self.latlon = None
        self.zoom_level = None
        self.bbox = None
        self.startPoint = None
        self.endPoint = None
        self.features = None
        self.shelterIndex = None
        self.OSGridReference = None
        self.tempWind = None
        self.debug = True
        self.numberOfPoints = 100

    def updateOptimiser(self, latlon, zoom_level, bbox, features, preferences):
        self.latlon = latlon
        self.zoom_level = float(zoom_level)
        self.bbox = self.getBBoxList(bbox)
        self.features = features
        self.preferences = self.updatePreferences(preferences)
        self.shelterIndex = self.getShelterIndex()
        self.OSGridReference = self.getOSGridReference()
        self.tempWind = self.getTempWind()
        # self.printStats()

        # self.convertToJson(
        #     get_min_path(self.bbox[0], self.bbox[1], math.floor(self.zoom_level))
        # )
        # print(self.minPathToPoint, flush=True)

    def make_heatmap(self):
        heatmap = heatmap_layer(self.bbox)
        heatmap.make_layers()

        x = heatmap.grid[0]
        y = heatmap.grid[1]
        z = heatmap.grid[2]

        n_spots = self.numberOfPoints
        grid_spots = np.concatenate(
            (
                np.array(
                    [
                        x[
                            np.unravel_index(
                                np.argsort(z.flatten())[-n_spots:], z.shape
                            )[0],
                            0,
                        ]
                    ]
                ).T,
                np.array(
                    [
                        y[
                            0,
                            np.unravel_index(
                                np.argsort(z.flatten())[-n_spots:], z.shape
                            )[1],
                        ]
                    ]
                ).T,
            ),
            1,
        )

        latlong_spots = []
        for i in range(grid_spots.shape[0]):
            latlong = grid2latlong(
                str(OSGridReference(grid_spots[i][0], grid_spots[i][1]))
            )
            latlong_spots.append([latlong.longitude, latlong.latitude])

        heatmap.plot_heatmap()
        return latlong_spots

    def convertToJson(self, minPath):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": minPath},
                }
            ],
        }
        output = open("minPath.geojson", "w")
        json.dump(geojson, output)
        return geojson

    def getBBoxList(self, bbox):
        bboxLatLon = re.findall("\(.*?\)", bbox)
        bboxList = []
        for latLon in bboxLatLon:
            bboxList.append(
                latLon.replace("(", "").replace(")", "").replace(" ", "").split(",")
            )
        bbox = [
            [float(bboxList[0][0]), float(bboxList[0][1])],
            [float(bboxList[1][0]), float(bboxList[1][1])],
        ]
        return bbox

    def setPoint(self, start_latlonDict, end_latlonDict):

        self.startPoint = [start_latlonDict["lng"], start_latlonDict["lat"]]
        self.endPoint = [end_latlonDict["lng"], end_latlonDict["lat"]]

        # if self.startPoint != None and self.endPoint != None:
        min_path = new_get_min_path(self.startPoint, self.endPoint, 14)

        print("\n\nMIN PATH: \n\n", min_path, "\n\n")

        return self.convertToJson(min_path)

    def getFeatures(self):
        pass
        # print(self.features)

    def getShelterIndex(self):
        shelterIndex = wind_shelter(
            self.latlon["lat"], self.latlon["lng"], math.ceil(float(self.zoom_level))
        )
        return shelterIndex

    def getOSGridReference(self):
        return str(latlong2grid(self.latlon["lat"], self.latlon["lng"]))

    def getTempWind(self):
        get_weather = weather_mesh([self.latlon["lat"]], [self.latlon["lng"]])
        tempWind = get_weather["features"][0]["properties"]
        return tempWind

    def updatePreferences(self, newPreferences):
        preferences = {}
        keys = list(self.preferences.keys())
        prefList = []
        for preference in newPreferences:
            if preference.isdigit():
                prefList.append(preference)

        for i in range(0, len(prefList)):
            preferences[keys[i]] = prefList[i]
        return preferences

    def printStats(self):
        print("Latlon:", self.latlon, flush=True)
        print("zoom_level:", self.zoom_level, flush=True)
        print("bbox:", self.bbox, flush=True)
        print("shelterIndex:", self.shelterIndex, flush=True)
        print("OSGridReference:", self.OSGridReference, flush=True)
        print("preferences:", self.preferences, flush=True)


@app.route("/")
def home():
    return render_template("bivouac.html")


# @app.route("/start_destination", methods=["POST", "GET"])
# def start_destination():
#     if request.method == "POST":

#         start_location = json.loads(re.findall("\{.*?\}", request.form["location"])[1])

#         # minpath = optimiser.setPoint(start_location, "start")
#         # print("start minpath:\n", minpath, "\n")

#         data = {"status": "success", "start_location": start_location}

#         # print("start_destination coords:\n", type(location[1]), "\n")

#     return data, 200


@app.route("/end_destination", methods=["POST", "GET"])
def end_destination():
    if request.method == "POST":

        start_location = json.loads(
            re.findall("\{.*?\}", request.form["start_location"])[1]
        )
        end_location = json.loads(
            re.findall("\{.*?\}", request.form["end_location"])[1]
        )

        startPoint = [start_location["lng"], start_location["lat"]]
        endPoint = [end_location["lng"], end_location["lat"]]

        print("startPoint: ", startPoint)
        print("startPoint type: ", type(startPoint))

        print("endPoint: ", endPoint)
        print("endPoint type: ", type(endPoint))

        # if self.startPoint != None and self.endPoint != None:
        min_path = new_get_min_path(startPoint, endPoint, 13)

        minpath = optimiser.convertToJson(min_path)
        print("end MINPATHDSbujgibdufgld ghiuolb io:\n", minpath, "\n")

        # switches lat and long (changed it in setPoint instead)
        # minpath["features"][0]["geometry"]["coordinates"][:] = map(
        #     lambda l: list(reversed(l)),
        #     minpath["features"][0]["geometry"]["coordinates"],
        # )
        # print(minpath, flush=True)

        data = {"status": "success", "minpath": minpath}

        print("end_dest function minpath obtained!\n")

    return data, 200


@app.route("/create_heatmap", methods=["POST", "GET"])
def create_heatmap():
    if request.method == "POST":
        location = request.form["location"]
        best_points = optimiser.convertToJson(optimiser.make_heatmap())
        # best_points = optimiser.convertToJson([-2.602678,51.455691])
        data = {"status": "success", "points": best_points}
    return data, 200


@app.route("/set_preferences", methods=["POST", "GET"])
def get_preferences():
    if request.method == "POST":
        preferences = request.form["preferences"]
        data = {"status": "success"}
        try:
            optimiser.preferences = optimiser.updatePreferences(preferences)
        except NameError:
            pass
        # print(preferences, flush=True)

    return data, 200


@app.route("/get_result", methods=["POST", "GET"])
def process_result():
    if request.method == "POST":

        mouse_pos = request.form["mouse_info"]
        zoom_level = request.form["zoom_level"]
        bbox = request.form["bbox"]
        preferences = request.form["vals"]
        features = request.form["features"]

        # print(features)
        with open("data.geojson", "w") as f:
            json.dump(json.loads(features), f)

        latlon = json.loads(re.findall("\{.*?\}", mouse_pos)[1])
        optimiser.updateOptimiser(
            latlon, zoom_level, bbox, json.loads(features), preferences
        )
        # optimiser.make_heatmap()
        # print("Output :" + mouse_pos, flush=True)
        # print("Zoom level :" + zoom_level, flush=True)
        # print("Features :" + features, flush=True)
        # Define loads of interesting things here, ie list of coords to plot that is a path
        num_features = get_num_features(features)
        # add whatever keys and values we want to this
        data = {
            "status": "success",
            "some": num_features,
            "temp": optimiser.tempWind["Temp"],
            "wind": optimiser.tempWind["Wind"],
            "wind_shelter": round(optimiser.shelterIndex, 4),
            "osGrid": optimiser.OSGridReference,
        }
        # creating elevation matrix (needs to be using the bbox and latlon centre)
        MAPBOX_TOKEN = "pk.eyJ1IjoiY3Jpc3BpYW5tIiwiYSI6ImNsMG1oazJhejE0YzAzZHVvd2Z1Zjlhb2YifQ.cv0zlPYY6WnoKM9YLD1lMQ"
        # tile_coords = mercantile.tile(bbox[0][0], bbox[0][1], zoom_level)
        """tile_coords = mercantile.tile(bbox[0][0], bbox[0][1], zoom_level)
        upper_left = mercantile.ul(tile_coords)
        lnglat_mat = construct_lng_lat_matrix(upper_left, zoom_level)
        elevation_mat = getElevationMatrix(MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y)
        slope_mat = getSlopeMatrix(elevation_mat)"""

    return data, 200  # 200 tells ajax "success!"


def get_num_features(feats):
    dictionary = json.loads(feats)
    num = len(dictionary)
    return num


if __name__ == "__main__":
    global optimiser
    optimiser = Optimiser()
    app.run()
