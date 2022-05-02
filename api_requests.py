# coding:utf8
#%%
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import overpy
import json

#%%


def get_query_response(query_str):
    """
    :param query_str: String representing Overpass query
    :return: JSON response
    """
    api = overpy.Overpass()
    response = api.query(query_str)
    return response


def get_biergarten_data():
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
    response = requests.get(overpass_url, params={"data": overpass_query})

    data_dict = json.loads(response.text)

    return data_dict


def get_biergarten_coords(data):
    # Collect coords into list
    coords = []
    for element in data["elements"]:
        if element["type"] == "node":
            lon = element["lon"]
            lat = element["lat"]
            coords.append((lon, lat))
        elif "center" in element:
            lon = element["center"]["lon"]
            lat = element["center"]["lat"]
            coords.append((lon, lat))
    # Convert coordinates into numpy array
    X = np.array(coords)
    return X


def plot_coords(X):
    plt.plot(X[:, 0], X[:, 1], "o")
    plt.title("Biergarten in Germany")
    plt.xlim((6, 16))
    plt.ylim((47, 57))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.show()


def plot_biergarten_data_overpy(data):
    api = overpy.Overpass()
    r = api.query(
        """
    area["ISO3166-1"="DE"][admin_level=2];
    (node["amenity"="biergarten"](area);
     way["amenity"="biergarten"](area);
     rel["amenity"="biergarten"](area);
    );
    out center;
    """
    )
    coords = []
    coords += [(float(node.lon), float(node.lat)) for node in r.nodes]
    coords += [(float(way.center_lon), float(way.center_lat)) for way in r.ways]
    coords += [(float(rel.center_lon), float(rel.center_lat)) for rel in r.relations]


#%%
data = get_biergarten_data()
#%%
X = get_biergarten_coords(data)
#%%
plot_coords(X)
#%%
plot_biergarten_data_overpy(data)
#%%
data

#%%
query = """
node(1694620775);
complete(100) { nwr[amenity=pub](around:500); };
out center;"""
pub_crawl = get_query_response(query)
#%%
pub_crawl.nodes
#%%

a = np.random.rand(10000, 100000)
np.argmin(a)
