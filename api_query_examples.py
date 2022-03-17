#coding utf8
import overpy


def get_query_response(query_str):
    """
    :param query_str: String representing Overpass query
    :return: JSON response
    """
    api = overpy.Overpass()
    response = api.query(query_str)
    return response


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