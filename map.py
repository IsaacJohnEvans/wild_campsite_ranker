from flask import Flask, render_template, url_for, jsonify, request, json
import re
import math
app = Flask(__name__)
from basic_weather_calls import weather_mesh
from wind_shelter import wind_shelter
from OSGridConverter import latlong2grid

@app.route('/')
def home():
    return render_template('bivouac.html')


@app.route('/get_result', methods=['POST', 'GET'])
def process_result():
    if request.method == 'POST':
        mouse_pos = request.form['mouse_info']
        zoom_level = request.form['zoom_level']
        bbox = request.form['bbox']
        bboxList = getBBoxList(bbox)
        print(bboxList, flush= True)
        features = request.form['features']
        # print(features)
        with open('data.geojson', 'w') as f:
            json.dump(json.loads(features), f)

        latlon = json.loads(re.findall('\{.*?\}',mouse_pos)[1])

        get_weather = weather_mesh([latlon['lat']], [latlon['lng']])
        tempWind = get_weather['features'][0]['properties']
        osGrid = str(latlong2grid(latlon['lat'], latlon['lng']))

        shelter = wind_shelter(latlon['lat'], latlon['lng'], math.ceil(float(zoom_level)))

        
        
        # print("Output :" + mouse_pos, flush=True)
        # print("Zoom level :" + zoom_level, flush=True)
        # print("Features :" + features, flush=True)

        # Define loads of interesting things here, ie list of coords to plot that is a path
        num_features = get_num_features(features)

        # add whatever keys and values we want to this
        data = {"status": "success",
            "some": num_features,
            "temp": tempWind['Temp'],
            "wind": tempWind['Wind'],
            "wind_shelter": shelter,
            "osGrid": osGrid
            } 
        

    return data, 200 # 200 tells ajax "success!"
   
def get_num_features(feats):
    dictionary = json.loads(feats)
    num = len(dictionary)
    return num

def getBBoxList(bbox):

    bboxLatLon = re.findall('\(.*?\)', bbox)
    bboxList = []
    for latLon in bboxLatLon:
        bboxList.append(latLon.replace('(','').replace(')','').replace(' ','').split(','))

    return bboxList


if __name__ == "__main__":
    app.run(debug=True)