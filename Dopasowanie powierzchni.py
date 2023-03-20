import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


points = []
with open('simulate_clouds.xyz', 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        points.extend(row)

points = [x for x in zip(*[iter(points)]*3)]
points = np.array(points, dtype=float)
x, y, z = zip(*points)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
plt.title('points clouds in 3D', fontsize=14)
plt.tight_layout
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)
plt.show()
