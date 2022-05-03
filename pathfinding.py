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


def get_tile(lat, lng, zoom_level):
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
    djikstra(
        padded_mat,
        startNode=(1, 1),
        targetNode=(248, 250),
        zoomlevel=zoom_level,
        latitude=lat,
        elevation_multiplier=10,
        show_plot=True,
    )


def construct_lng_lat_matrix(ul, zoomlevel):

    matrix = np.zeros([256, 256], list)

    for i in range(256):
        for j in range(256):
            lng_lat = coord_to_lng_lat(ul, coord=(i, j), zoomlevel=zoomlevel)
            matrix[i, j] = [lng_lat[0], lng_lat[1]]

    return matrix


def construct_lng_lat_matrix2(tile):

    bbox = mercantile.bounds(tile)
    # Unpack upper-left lng lat
    ul_lng = bbox.west
    ul_lat = bbox.north
    # Unpack lower-right lng lat
    lr_lng = bbox.east
    lr_lat = bbox.south

    # print("ul", ul_lng, ul_lat)
    # print("lr", lr_lng, lr_lat)

    # delta lng & lat from upper-left to lower-right
    delta_lng = abs(bbox.east - bbox.west)
    delta_lat = abs(bbox.north - bbox.south)

    matrix = np.zeros([256, 256], list)

    for i in range(256):
        for j in range(256):
            cell_lng = ul_lng + (j * delta_lng) / 256
            cell_lat = ul_lat - (i * delta_lat) / 256
            matrix[i, j] = [cell_lng, cell_lat]

    return matrix


def coord_to_lng_lat(ul, coord, zoomlevel):
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


def lng_lat_to_coord(lng_lat_matrix, lng_lat):

    distances_matrix = np.zeros(
        [lng_lat_matrix.shape[0], lng_lat_matrix.shape[1]], dtype=float
    )
    for i in range(distances_matrix.shape[0]):
        for j in range(distances_matrix.shape[1]):
            distance = abs(sum(np.array(lng_lat_matrix[i, j]) - np.array(lng_lat)))
            distances_matrix[i, j] = distance

    min_idx = np.unravel_index(distances_matrix.argmin(), distances_matrix.shape)

    return min_idx


def djikstra(matrix, startNode, targetNode, resolution, elevation_multiplier=4):

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


def get_min_path(
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

    upper_left = mercantile.ul(tile_coords)

    # lng_lat_matrix = construct_lng_lat_matrix(upper_left, zoomlevel=zoom)
    lng_lat_matrix = construct_lng_lat_matrix2(tile_coords)

    startNode = lng_lat_to_coord(lng_lat_matrix, lng_lat=list(end_lng_lat))
    targetNode = lng_lat_to_coord(lng_lat_matrix, lng_lat=list(end_lng_lat))
    # print("startNode", startNode)
    # print("targetNode", targetNode)
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

    lng_lat_matrix = construct_lng_lat_matrix2(tile_coords)
    startNode = lng_lat_to_coord(lng_lat_matrix, start_lng_lat)
    targetNode = lng_lat_to_coord(lng_lat_matrix, end_lng_lat)

    # Generate the shortest path as a sequence of lng, lat tuples
    node_path = djikstra(
        padded_mat,
        startNode=startNode,
        targetNode=(targetNode[0], targetNode[1]),
        resolution=resolution,
        elevation_multiplier=elevation_multiplier,
    )

    # Get lng and lat of upper-left of tile
    upper_left = mercantile.ul(tile_coords)
    # Gets path as series of longitude and latitude coordinates
    lnglatPath = [coord_to_lng_lat(upper_left, coord, zoom) for coord in node_path]

    lnglatPath = lnglatPath[::-1]

    if show_img:
        plt.imshow(elevation_mat, interpolation="nearest")
        xs = [x[0] for x in node_path]
        ys = [x[1] for x in node_path]
        plt.plot(xs, ys, "r-")
        plt.show()

    return lnglatPath


def get_min_path_from_bbox(bbox, elevation_multiplier=10, show_img=False):
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
    lng_lat_mat = construct_lng_lat_matrix(upper_left, tile_coords.z)

    # Pad matrix with infinities to represent boundaries
    padded_mat = np.pad(
        elevation_mat, [(1, 1), (1, 1)], mode="constant", constant_values=np.Inf
    )

    # resolution = 156543.03 meters/pixel * cos(latitude) / (2 ^ zoomlevel)
    latitude_radians = bbox[1] * math.pi / 180
    resolution = abs(156543.03 * np.cos(latitude_radians) / (2**tile_coords.z))

    startNode = lng_lat_to_coord(lng_lat_mat, [bbox[0], bbox[1]])
    targetNode = lng_lat_to_coord(lng_lat_mat, [bbox[2], bbox[3]])

    # Generate the shortest path as a sequence of lng, lat tuples
    node_path = djikstra(
        padded_mat,
        startNode=(startNode[0] + 1, startNode[1] + 1),
        targetNode=(targetNode[0] + 1, targetNode[1] + 1),
        resolution=resolution,
        elevation_multiplier=elevation_multiplier,
    )

    # Gets path as series of longitude and latitude coordinates
    lnglatPath = [
        coord_to_lng_lat(upper_left, coord, tile_coords.z) for coord in node_path
    ]

    if show_img:
        plt.imshow(elevation_mat, interpolation="nearest")
        xs = [x[0] for x in node_path]
        ys = [x[1] for x in node_path]
        plt.plot(xs, ys, "r-")
        plt.show()

    return lnglatPath
