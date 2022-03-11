#coding:utf8
#%%
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import overpy
import json

'''
data['elements'][0]['lat']
#[data.get(key) for key in data.keys()]
data.keys()
data_list = data['elements']
for i in data_list:
    data_array = 
print(type(data))
data_json = json.dumps(data, indent = 5)
with open('beer_garden_data.json', "w") as outfile:
    json.dump(data_json, outfile)
'''
def get_data():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    area["ISO3166-1"="DE"][admin_level=2];
    (node["amenity"="biergarten"](area);
     way["amenity"="biergarten"](area);
     rel["amenity"="biergarten"](area);
    );
    out center;
    """
    response = requests.get(overpass_url,
                            params={'data': overpass_query})

    data_dict = json.loads(response.text)
    return data_dict

def get_coords(data):
    # Collect coords into list
    coords = []
    for element in data['elements']:
        if element['type'] == 'node':
            lon = element['lon']
            lat = element['lat']
            coords.append((lon, lat))
        elif 'center' in element:
            lon = element['center']['lon']
            lat = element['center']['lat']
            coords.append((lon, lat))
    # Convert coordinates into numpy array
    X = np.array(coords)
    return X

