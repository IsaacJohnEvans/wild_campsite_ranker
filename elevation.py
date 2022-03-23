import requests
import ast
from local_config import MAPBOX_TOKEN
from PIL import Image
import io
import numpy as np


def getFeatureData(lng, lat, ACCESS_TOKEN):
    # Construct the API request.
    url = f'https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/{lng},{lat}.json?layers=contour&limit=50&access_token={ACCESS_TOKEN}'

    response = requests.get(url)

    if response.status_code == 200:
        print("Error: Empty FeatureCollection.")
        return

    data_raw = response.content
    data_dict = ast.literal_eval(data_raw)

    print(data_dict)

    return data_dict


def getRasterRGB(ACCESS_TOKEN, zoom=14, x=3826, y=6127):
    """Gets RGBA RAWPNG raster elevation heatmap using tilename values x & y at specified zoom level"""
    # need to convert long and lat to x and y using slippy map
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{x}/{y}.pngraw?access_token={ACCESS_TOKEN}"
    response = requests.get(url)
    #print(response.content)
    return response.content


def getRasterDEM(ACCESS_TOKEN):
    url = f"https://api.mapbox.com/raster/v1/mapbox.mapbox-terrain-dem-v1/14/3826/6127.pngraw?sku=sk.eyJ1IjoiY3Jpc3BpYW5tIiwiYSI6ImNsMG1oeGJqejA1a3kzYm83bGhzZGFzbm0ifQ.Ncne-l2qXx2arHsHpUmVcw&access_token={ACCESS_TOKEN}"
    response = requests.get(url)
    #print(response.content)
    return response.content


def rasterToImage(raster):
    """Convert rawpng rgba octets into PIL image"""
    imagePIL = Image.open(io.BytesIO(raster))
    imagePIL.show()
    return imagePIL


def getElevationMatrix(img):
    """Generates matrix of elevation values for each RGBA raster pixel"""
    elevation_matrix = np.ones( (256, 256), dtype=float)
    for i in range(256):
        for j in range(256):
            coord = x, y = i, j
            pixel_rgba = img.getpixel(coord)
            # elevation = -10000 + (({R} * 256 * 256 + {G} * 256 + {B}) * 0.1)
            elevation = -10000 + ((pixel_rgba[0] * 256 * 256 + pixel_rgba[1] * 256 + pixel_rgba[2])) * 0.1
            elevation_matrix[i,j] = elevation
    return elevation_matrix

