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

ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
plt.title('points clouds in 3D', fontsize=14)
plt.tight_layout()
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)


clusterer = KMeans(n_clusters=3)
clusterer.fit(points)
y_pred = clusterer.predict(points)
z_pred = clusterer.predict(points)

red = y_pred == 0
blue = y_pred == 1
cyan = y_pred == 2

red_z = z_pred == 0
blue_z = z_pred == 1
cyan_z = z_pred == 2

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(points[red, 0], points[red, 1], points[red_z, 2], c='red')
ax.scatter3D(points[blue, 0], points[blue, 1], points[blue_z, 2], c='blue')
ax.scatter3D(points[cyan, 0], points[cyan, 1], points[cyan_z, 2], c='cyan')


plt.show()
