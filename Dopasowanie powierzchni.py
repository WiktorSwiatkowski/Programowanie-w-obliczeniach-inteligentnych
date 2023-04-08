import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from copy import copy
from numpy.random import default_rng
import random


random.seed(300)
np.random.seed(300)

points = []
with open('simulate_clouds.xyz', 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        points.extend(row)

points = [x for x in zip(*[iter(points)]*3)]
points = np.array(points, dtype=float)
X, Y, Z = zip(*points)

ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z)
plt.title('Chmury punktow', fontsize=16)
plt.tight_layout()
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
ax.set_zlabel('z', fontsize=10)

clusterer = KMeans(n_clusters=3)
clusterer.fit(points)
y_pred = clusterer.predict(points)

green = y_pred == 0
magenta = y_pred == 1
brown = y_pred == 2

#plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(points[green, 0], points[green, 1], points[green, 2], c='green')
ax.scatter3D(points[magenta, 0], points[magenta, 1], points[magenta, 2], c='magenta')
ax.scatter3D(points[brown, 0], points[brown, 1], points[brown, 2], c='brown')
plt.title('Klastry', fontsize=16)
plt.tight_layout()
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
ax.set_zlabel('z', fontsize=10)


rng = default_rng()


class RANSAC:
    def __init__(self, n=20, k=100, t=20, d=20, model=None, loss=None, metric=None):
        self.n = n
        self.k = k
        self.t = t
        self.d = d
        self.model = model
        self.loss = loss
        self.metric = metric
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers])

            thresholded = (
                self.loss(X[ids][self.n :, 2], maybe_model.predict(X[ids][self.n :]))
                < self.t
            )

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(X[inlier_points])

                this_error = self.metric(
                    X[inlier_points][:, 2], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X)[:, np.newaxis]


class PlaneRegression:
    def __init__(self):
        self.coeffs = None

    def fit(self, points):

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        A = np.c_[x, y, np.ones_like(x)]

        self.coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return self

    def predict(self, point):

        x, y = point[:,0], point[:,1]

        z = self.coeffs[0] * x + self.coeffs[1] * y + self.coeffs[2]

        return z


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


if __name__ == "__main__":

    regressor = RANSAC(model=PlaneRegression(), loss=square_error_loss, metric=mean_square_error)

    regressor.fit(points[green])
    plt.style.use("seaborn-darkgrid")
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[green, 0], points[green, 1], points[green, 2], c='green', alpha=0.1)
    ax.scatter3D(points[green,0], points[green,1], regressor.predict(points[green,0:2]), color="red")

    regressor.fit(points[magenta])
    plt.style.use("seaborn-darkgrid")
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[magenta, 0], points[magenta, 1], points[magenta, 2], c='magenta', alpha=0.1)
    ax.scatter3D(points[magenta, 0], points[magenta, 1], regressor.predict(points[magenta, 0:2]), color="red")

    regressor.fit(points[brown])
    plt.style.use("seaborn-darkgrid")
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[brown, 0], points[brown, 1], points[brown, 2], c='brown', alpha=0.1)
    ax.scatter3D(points[brown, 0], points[brown, 1], regressor.predict(points[brown, 0:2]), color="red")

plt.show()
