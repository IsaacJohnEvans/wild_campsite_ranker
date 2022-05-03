import numpy as np
import mercantile
import matplotlib.pyplot as plt
from elevation import getElevationMatrix, rasterToImage, getRasterRGB

# from local_config import MAPBOX_TOKEN
import math

MAPBOX_TOKEN = "pk.eyJ1IjoiY3Jpc3BpYW5tIiwiYSI6ImNsMG1oazJhejE0YzAzZHVvd2Z1Zjlhb2YifQ.cv0zlPYY6WnoKM9YLD1lMQ"

"""tile_coords = mercantile.tile(lng=-95.9326171875, lat=41.26129149391987, zoom=12)
print(tile_coords)
elevation_mat = getElevationMatrix(MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y)
padded_mat = np.pad(elevation_mat, [(1, 1), (1, 1)], mode='constant', constant_values=np.Inf)
print(padded_mat)
# Get latitude and longitude at upper-left of tile
upper_left = mercantile.ul(tile_coords)
print("upperleft:", upper_left)"""


def new_get_tile(lat, lng, zoom_level):
    tile_coords = mercantile.tile(lng=lng, lat=lat, zoom=zoom_level)
    elevation_mat = getElevationMatrix(
        MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y
    )
    padded_mat = np.pad(
        elevation_mat, [(1, 1), (1, 1)], mode="constant", constant_values=np.Inf
    )
    # print(padded_mat, flush=True)
    # Get latitude and longitude at upper-left of tile
    upper_left = mercantile.ul(tile_coords)
    new_djikstra(
        padded_mat,
        startNode=(1, 1),
        targetNode=(248, 250),
        zoomlevel=zoom_level,
        latitude=lat,
        elevation_multiplier=10,
        show_plot=True,
    )


def new_construct_lng_lat_matrix(ul, zoomlevel):

    matrix = np.zeros([256, 256], list)

    for i in range(256):
        for j in range(256):
            lng_lat = new_coord_to_lng_lat(ul, coord=(i, j), zoomlevel=zoomlevel)
            matrix[i, j] = [lng_lat[0], lng_lat[1]]

    return matrix


def new_construct_lng_lat_matrix2(tile):

    bbox = mercantile.bounds(tile)
    # Unpack upper-left lng lat
    ul_lng = round(bbox.west, 7)
    ul_lat = round(bbox.north, 7)
    # Unpack lower-right lng lat
    lr_lng = round(bbox.east, 7)
    lr_lat = round(bbox.south, 7)

    # delta lng & lat from upper-left to lower-right
    delta_lng = round(abs(10000000 * bbox.east - 10000000 * bbox.west), 7)
    delta_lat = round(abs(10000000 * bbox.north - 10000000 * bbox.south), 7)

    matrix = np.zeros([256, 256], list)

    for i in range(256):
        for j in range(256):
            cell_lng = round(10000000 * ul_lng + (j * delta_lng) / 256, 7)
            cell_lat = round(10000000 * ul_lat - (i * delta_lat) / 256, 7)
            matrix[i, j] = [
                round(cell_lng / 10000000, 7),
                round(cell_lat / 10000000, 7),
            ]

    return matrix


def new_coord_to_lng_lat2(ul, coord, zoomlevel):
    """Converts x,y matrix coordinate into longitude and latitude coordinates"""
    # Unpack upper-left of tile longitude and latitude
    ul_lat = ul.lat
    ul_lng = ul.lng

    # Calculate distance between each pixel
    latitude_radians = ul_lat * math.pi / 180
    resolution = abs(156543.03 * np.cos(latitude_radians) / (2**zoomlevel))

    # Radius of Earth in metres
    R = 6378137

    # Change in distance (delta pixels * resolution in metres)
    dn = coord[0] * resolution
    de = coord[1] * resolution

    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * ul_lat / 180))

    latO = ul_lat + dLat * 180 / math.pi
    lonO = ul_lng + dLon * 180 / math.pi

    return [lonO, latO]


def new_coord_to_lng_lat(coord, tile):

    bbox = mercantile.bounds(tile)
    # Unpack upper-left lng lat
    ul_lng = bbox.west
    ul_lat = bbox.north
    # Unpack lower-right lng lat
    lr_lng = bbox.east
    lr_lat = bbox.south

    # delta lng & lat from upper-left to lower-right
    total_delta_lng = abs(abs(lr_lng) - abs(ul_lng))
    total_delta_lat = abs(abs(lr_lat) - abs(ul_lat))

    lng = ul_lng + (coord[0] / 255) * total_delta_lng
    lat = ul_lat - (coord[1] / 255) * total_delta_lat

    return [lng, lat]


def new_lng_lat_to_coord(lng_lat, tile):

    target_lng = lng_lat[0]
    target_lat = lng_lat[1]

    bbox = mercantile.bounds(tile)
    # Unpack upper-left lng lat
    ul_lng = bbox.west
    ul_lat = bbox.north
    # Unpack lower-right lng lat
    lr_lng = bbox.east
    lr_lat = bbox.south

    # delta lng & lat from upper-left to lower-right
    total_delta_lng = abs(abs(lr_lng) - abs(ul_lng))
    total_delta_lat = abs(abs(lr_lat) - abs(ul_lat))
    target_delta_lng = abs(abs(target_lng) - abs(ul_lng))
    target_delta_lat = abs(abs(target_lat) - abs(ul_lat))

    x = target_delta_lng / total_delta_lng
    y = target_delta_lat / total_delta_lat

    x = math.floor(x * 255)
    y = math.floor(y * 255)
    return [x, y]


def new_djikstra(matrix, startNode, targetNode, resolution, elevation_multiplier=4):

    neighbourDiffs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
    visitedNodes = {
        startNode: 0
    }  # Dictionary of nodes and their shortest discovered cumulative distance
    frontierNodes = dict()
    parentDict = dict()

    currentNode = startNode
    currentDist = 0
    # print("startNode:", startNode)
    # print("targetNode:", targetNode)
    while True:

        neighbourNodes = set(
            tuple(np.array(currentNode) + np.array(diff)) for diff in neighbourDiffs
        )

        for node in neighbourNodes:
            if node not in visitedNodes.keys():
                # Generate weighting for traversing to neighbouring node
                neighbourDist = (
                    currentDist
                    + resolution
                    + elevation_multiplier
                    * abs((float(matrix[node]) - float(matrix[currentNode])))
                )

                # Update frontier if newly-discovered distance is smaller than existing frontier distance
                try:
                    if neighbourDist < frontierNodes[node]:
                        frontierNodes[node] = neighbourDist
                        # Update parent node for shortest path
                        parentDict[node] = currentNode
                except KeyError:
                    frontierNodes[node] = neighbourDist
                    parentDict[node] = currentNode

        # Search frontier nodes for smallest distance
        smallestFrontierNode = min(frontierNodes, key=frontierNodes.get)

        # Change current node to smallest frontier node
        currentNode = smallestFrontierNode
        currentDist = frontierNodes[currentNode]

        # Remove new current node from frontier
        frontierNodes.pop(currentNode, None)

        # Add new current node to visited nodes
        visitedNodes[currentNode] = currentDist
        if targetNode in visitedNodes.keys():
            # print("DONE")
            break

    # print(len(visitedNodes))

    # Backtracking to get path of nodes
    currentNode = targetNode
    nodePath = [currentNode]
    while currentNode != startNode:
        currentNode = parentDict[currentNode]
        nodePath.append(currentNode)
    # print(nodePath)

    return nodePath


def new_get_min_path(
    start_lng_lat, end_lng_lat, zoom, elevation_multiplier=5, show_img=False
):

    # Get direction of end location relative to start
    x_delta = end_lng_lat[0] - start_lng_lat[0]
    y_delta = end_lng_lat[1] - start_lng_lat[1]

    if (x_delta < 0) and (y_delta > 0):
        tile_lng_lat = end_lng_lat
        # startNode = (256, 256)

    elif (x_delta < 0) and (y_delta < 0):
        tile_lng_lat = (end_lng_lat[0], start_lng_lat[1])
        # startNode = (1, 256)

    elif (x_delta > 0) and (y_delta < 0):
        tile_lng_lat = start_lng_lat
        # startNode = (1, 1)

    elif (x_delta > 0) and (y_delta > 0):
        tile_lng_lat = (start_lng_lat[0], end_lng_lat[1])
        # startNode = (256, 1)

    # Get mercantile tile x,y,z from lng, lat, zoom
    tile_coords = mercantile.tile(lng=tile_lng_lat[0], lat=tile_lng_lat[1], zoom=zoom)

    # Get elevation matrix
    elevation_mat = getElevationMatrix(
        MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y
    )
    # Pad matrix with infinities to represent boundaries
    padded_mat = np.pad(
        elevation_mat, [(1, 1), (1, 1)], mode="constant", constant_values=np.Inf
    )

    # resolution = 156543.03 meters/pixel * cos(latitude) / (2 ^ zoomlevel)
    latitude_radians = tile_lng_lat[1] * math.pi / 180
    resolution = abs(156543.03 * np.cos(latitude_radians) / (2**zoom))

    startNode = new_lng_lat_to_coord(start_lng_lat, tile_coords)
    targetNode = new_lng_lat_to_coord(end_lng_lat, tile_coords)

    # Generate the shortest path as a sequence of lng, lat tuples
    node_path = new_djikstra(
        padded_mat,
        startNode=(startNode[0], startNode[1]),
        targetNode=(targetNode[0], targetNode[1]),
        resolution=resolution,
        elevation_multiplier=elevation_multiplier,
    )

    # Gets path as series of longitude and latitude coordinates
    lnglatPath = [new_coord_to_lng_lat(coord, tile_coords) for coord in node_path]
    lnglatPath = lnglatPath[::-1]

    if show_img:
        plt.imshow(elevation_mat, interpolation="nearest")
        xs = [x[0] for x in node_path]
        ys = [x[1] for x in node_path]
        plt.plot(xs, ys, "r-")
        plt.show()

    return lnglatPath


def new_get_min_path_from_bbox(bbox, elevation_multiplier=10, show_img=False):
    if bbox[0] > bbox[2]:  # coord 1 more east
        if bbox[1] > bbox[3]:  # coord 1 more north
            bbox = [bbox[2], bbox[1], bbox[0], bbox[3]]
        else:  # coord 1 more south
            bbox = [bbox[2], bbox[3], bbox[0], bbox[1]]
    else:  # coord 1 more west
        if bbox[1] > bbox[3]:  # coord 1 more north
            pass
        else:  # coord 1 more south
            bbox = [bbox[0], bbox[3], bbox[2], bbox[1]]

    tile_coords = mercantile.bounding_tile(bbox[0], bbox[1], bbox[2], bbox[3])

    upper_left = mercantile.ul(tile_coords)

    elevation_mat = getElevationMatrix(
        MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y
    )
    lng_lat_mat = new_construct_lng_lat_matrix(upper_left, tile_coords.z)

    # Pad matrix with infinities to represent boundaries
    padded_mat = np.pad(
        elevation_mat, [(1, 1), (1, 1)], mode="constant", constant_values=np.Inf
    )

    # resolution = 156543.03 meters/pixel * cos(latitude) / (2 ^ zoomlevel)
    latitude_radians = bbox[1] * math.pi / 180
    resolution = abs(156543.03 * np.cos(latitude_radians) / (2**tile_coords.z))

    startNode = new_lng_lat_to_coord(lng_lat_mat, [bbox[0], bbox[1]])
    targetNode = new_lng_lat_to_coord(lng_lat_mat, [bbox[2], bbox[3]])

    # Generate the shortest path as a sequence of lng, lat tuples
    node_path = new_djikstra(
        padded_mat,
        startNode=(startNode[0] + 1, startNode[1] + 1),
        targetNode=(targetNode[0] + 1, targetNode[1] + 1),
        resolution=resolution,
        elevation_multiplier=elevation_multiplier,
    )

    # Gets path as series of longitude and latitude coordinates
    lnglatPath = [
        new_coord_to_lng_lat(upper_left, coord, tile_coords.z) for coord in node_path
    ]

    if show_img:
        plt.imshow(elevation_mat, interpolation="nearest")
        xs = [x[0] for x in node_path]
        ys = [x[1] for x in node_path]
        plt.plot(xs, ys, "r-")
        plt.show()

    return lnglatPath
