#coding : utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_class import map_feature, map_layer, heatmap_layer


distance = 2
n_points = 100
effect = 1
x_cen, y_cen = 362000, 326000
min_point = -1000
max_point = 1000
values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0])

heatmap = heatmap_layer()
heatmap.make_grid(x_cen, y_cen, n_points, min_point, max_point)
x, y, z = heatmap.grid
layer1 = map_layer(x, y, z, 'layer1', effect, distance, values)

layer1.get_features()
layer1.bool_features()
print(layer1.poly_bool.any())

layer1.draw_heatmap()
layer1.plot_heatmap()

'''
p = np.array([[361946, 326600], [361998, 326594], [362083, 326526], [362109, 326452], [362104, 326343], [362201, 326371], [362230, 326294], [362112, 326276], [362132, 326198], [362093, 326158], [362061, 326020], [362107, 325943], [362085, 325883], [362026, 325798], [362000, 325797], [361986, 325847], [362024, 325945], [361922, 326094], [361879, 326135], [361863, 326203], [361876, 326226], [361862, 326386], [361885, 326550], [361946, 326600]])

plt.figure(2)
plt.plot(p[:, 0], p[:, 1], '-o')
plt.show()
'''