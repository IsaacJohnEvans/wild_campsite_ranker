from api_requests import *


def uk_pubs():

    query = \
        """             [out:json];
                        (area["ISO3166-1"="GB"][admin_level=2];)->.a;
                        (node["amenity"="pub"](area);
                         way["amenity"="pub"](area);
                         rel["amenity"="pub"](area);
                        );
                        out center;"""

    response = get_query_response(query)
    print(response.nodes)


uk_pubs()