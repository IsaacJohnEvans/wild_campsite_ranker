import numpy as np
import mercantile
import matplotlib.pyplot as plt
from elevation import getElevationMatrix, rasterToImage, getRasterRGB
from local_config import MAPBOX_TOKEN
import math

tile_coords = mercantile.tile(lng=-95.9326171875, lat=41.26129149391987, zoom=10)
print(tile_coords)
elevation_mat = getElevationMatrix(MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y)
padded_mat = np.pad(elevation_mat, [(1, 1), (1, 1)], mode='constant', constant_values=np.Inf)
print(padded_mat)


def djikstra(matrix, startNode, targetNode, zoomlevel, latitude, elevation_multiplier=2):
    # resolution = 156543.03 meters/pixel * cos(latitude) / (2 ^ zoomlevel)
    print("latitiude:", latitude)
    latitude_radians = latitude * math.pi / 180
    resolution = abs(156543.03 * np.cos(latitude_radians) / (2 ** zoomlevel))
    print("resolution:", resolution)
    neighbourDiffs = [[0,1], [0,-1], [-1,0], [1,0]]
    visitedNodes = {startNode: 0} # Dictionary of nodes and their shortest discovered cumulative distance
    frontierNodes = dict()
    parentDict = dict()

    currentNode = startNode
    currentDist = 0

    while True:

        neighbourNodes = set(tuple(np.array(currentNode) + np.array(diff)) for diff in neighbourDiffs)

        for node in neighbourNodes:
            if node not in visitedNodes.keys():
                # Generate weighting for traversing to neighbouring node
                neighbourDist = currentDist + resolution + elevation_multiplier * abs((matrix[node] - matrix[currentNode]))

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
            print("DONE")
            break

    print(len(visitedNodes))

    # Backtracking to get path
    currentNode = targetNode
    nodePath = [currentNode]
    while currentNode != startNode:
        currentNode = parentDict[currentNode]
        nodePath.append(currentNode)
    print(nodePath)

    plt.imshow(elevation_mat, interpolation='nearest')
    xs = [x[0] for x in nodePath]
    ys = [x[1] for x in nodePath]
    plt.plot(xs, ys, 'r-')
    plt.show()


djikstra(padded_mat, startNode=(1,1), targetNode=(248, 250), zoomlevel=10, latitude=41.26129149391987, elevation_multiplier=10)
# scipy convolve 2D
