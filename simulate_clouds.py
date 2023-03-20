from scipy.stats import norm
from csv import writer
import numpy as np


def generate_points(num_points: int = 2000):
    distribution_x = norm(loc=-50, scale=15)
    distribution_y = norm(loc=0, scale=100)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


def generate_points_pion(num_points: int = 2000):
    distribution_x = norm(loc=-100, scale=0.05)
    distribution_y = norm(loc=0, scale=100)
    distribution_z = norm(loc=0.2, scale=20)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


def generate_points_cyl(radius=10, height=50, density=1000, turns=40):
    angle = np.linspace(0, turns * np.pi, density)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.linspace(0, height, num=density)

    points = zip(x, y, z)
    return points


if __name__ == '__main__':
    cloud_points = generate_points(2000)
    cloud_points_pion = generate_points_pion(2000)
    cylinder = generate_points_cyl()
    with open('simulate_clouds.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)

        for p in cloud_points:
            csvwriter.writerow(p)
        for p in cloud_points_pion:
            csvwriter.writerow(p)
        for p in cylinder:
            csvwriter.writerow(p)

