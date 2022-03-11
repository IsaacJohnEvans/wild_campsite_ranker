import json
import overpy


def get_query_response(query_str):
    """

    :param query_str: String representing Overpass query
    :return: JSON response
    """
    api = overpy.Overpass()
    response = api.query(query_str)

    return response

biergarten_reponse = get_query_response("""
                                            [out:json];
                                            area["ISO3166-1"="DE"][admin_level=2];
                                            (node["amenity"="biergarten"](area);
                                             way["amenity"="biergarten"](area);
                                             rel["amenity"="biergarten"](area);
                                            );
                                            out center;
                                            """)
print(biergarten_reponse.nodes)

