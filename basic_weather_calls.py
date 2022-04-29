
import requests


def weather_mesh(lat_list,lon_list):
    
    '''Basic function which outputs a dictionary in the format of a geojson.
    
    Input in the form of a list or numpy array of latitudes and longitudes.
    
    '''
    num_points = len(lat_list)
    feature_list=[]

    for i in range(num_points):
    
        call = 'https://api.openweathermap.org/data/2.5/weather?lat='+str(lat_list[i])+'&lon='+str(lon_list[i])+'&appid=e64e8059142aa7d4aa3126405fb4b4d2&units=metric'

        response = requests.get(call)

        data = response.json()
        feature_geojson = {
                "type": "Feature",
                "geometry" : {
                    "type": "Point",
                    "coordinates": [data['coord']["lon"], data['coord']["lat"]],
                    },
                "properties" :{ 
                    'Temp': data['main']['temp'],
                    'Wind' : data['wind']['speed']
                }
            }

        feature_list.append(feature_geojson)




    geojson = {
        "type": "FeatureCollection",
        "features": [i for i in feature_list

        ] 
    }

    return geojson


def wind_direction(lat,long):
    call = 'https://api.openweathermap.org/data/2.5/weather?lat='+str(lat)+'&lon='+str(long)+'&appid=e64e8059142aa7d4aa3126405fb4b4d2&units=metric'

    response = requests.get(call)
    data = response.json()
    
    return data['wind']['deg']
    
    
    