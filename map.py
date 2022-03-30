from flask import Flask, render_template, url_for, jsonify, request, json
import json, re
app = Flask(__name__)
from basic_weather_calls import weather_mesh


@app.route('/')
def home():
    return render_template('bivouac.html')


@app.route('/get_result', methods=['POST', 'GET'])
def process_result():
    if request.method == 'POST':
        mouse_pos = request.form['mouse_info']  
        zoom_level = request.form['zoom_level']
        features = request.form['features']
        latlon = json.loads(re.findall('\{.*?\}',mouse_pos)[1])

        get_weather = weather_mesh([latlon['lat']], [latlon['lng']])
        tempWind = get_weather['features'][0]['properties']

        with open('file.json', 'w') as f:
            json.dump(features, f)
        
        # print("Output :" + mouse_pos, flush=True)
        # print("Zoom level :" + zoom_level, flush=True)
        # print("Features :" + features, flush=True)

        # Define loads of interesting things here, ie list of coords to plot that is a path
        num_features = get_num_features(features)

        # add whatever keys and values we want to this
        data = {"status": "success",
            "some": num_features,
            "temp": tempWind['Temp'],
            "wind": tempWind['Wind']
            } 
        

    return data, 200 # 200 tells ajax "success!"
   
def get_num_features(feats):
    dictionary = json.loads(feats)
    num = len(dictionary)
    return num

if __name__ == "__main__":
    app.run(debug=True)