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


