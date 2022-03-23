import numpy as np
import matplotlib.pyplot as plt
from elevation import getElevationMatrix, rasterToImage, getRasterRGB
from local_config import MAPBOX_TOKEN

# raster = getRasterRGB(MAPBOX_TOKEN)
# image = rasterToImage(raster)
elevation_mat = getElevationMatrix(MAPBOX_TOKEN, 14, 3826, 6127)
padded_mat = np.pad(elevation_mat, [(1, 1), (1, 1)], mode='constant', constant_values=np.Inf)
print(padded_mat)


def djikstra(matrix, startNode):
    # resolution = 156543.03 meters/pixel * cos(latitude) / (2 ^ zoomlevel)
    latitude = 0
    zoomlevel = 14
    resolution = 156543.03 * np.cos(latitude) / (2 ^ zoomlevel)
    elevation_multiplier = 2
    targetNode = (248, 250)
    neighbourDiffs = [[0,1], [0,-1], [-1,0], [1,0]]
    visitedNodes = {startNode: 0} # Dictionary of nodes and their shortest discovered cumulative distance
    frontierNodes = dict()
    parentDict = dict()
    optimalParents = dict()

    currentNode = startNode
    currentDist = 0

    while True:

        neighbourNodes = set(tuple(np.array(currentNode) + np.array(diff)) for diff in neighbourDiffs)
        #print(neighbourNodes)

        neighbourDistances = dict()
        for node in neighbourNodes:
            if node not in visitedNodes.keys():
                # Generate weighting for traversing to neighbouring node
                neighbourDist = currentDist + resolution + elevation_multiplier * (matrix[node] - matrix[currentNode])

                # Update frontier if newly-discovered distance is smaller than existing frontier distance
                try:
                    if neighbourDist < frontierNodes[node]:
                        frontierNodes[node] = neighbourDist
                        # Update parent node for shortest path
                        parentDict[node] = currentNode
                except KeyError:
                    frontierNodes[node] = neighbourDist
                    parentDict[node] = currentNode

        #print(frontierNodes)

        # Search frontier nodes for smallest distance
        smallestFrontierNode = min(frontierNodes, key=frontierNodes.get)
        #print(smallestFrontierNode)

        # Change current node to smallest frontier node
        currentNode = smallestFrontierNode
        currentDist = frontierNodes[currentNode]

        # Remove new current node from frontier
        frontierNodes.pop(currentNode, None)

        # Add new current node to visited nodes
        visitedNodes[currentNode] = currentDist
        optimalParents[currentNode] = parentDict
        #print(currentNode)
        if targetNode in visitedNodes.keys():
            print("DONE")
            break

    #print(parentDict)
    print(len(visitedNodes))

    # Backtracking to get path
    currentNode = targetNode
    nodePath = [currentNode]
    while currentNode != startNode:
        currentNode = parentDict[currentNode]
        nodePath.append(currentNode)
    #nodePath = nodePath.reverse()
    print(nodePath)

    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    xs = [x[0] for x in nodePath]
    ys = [x[1] for x in nodePath]
    plt.plot(xs, ys)
    plt.show()


djikstra(padded_mat, startNode=(1,1))
