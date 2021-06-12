import numpy as np
import math
from sklearn import linear_model
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def findPlane(pcd):
    xyz = np.asarray(pcd.points)
    xy = xyz[:, :2]
    z = xyz[:, 2]
    ransac = linear_model.RANSACRegressor(residual_threshold=0.01)
    ransac.fit(xy, z)
    a, b = ransac.estimator_.coef_  # coefficients
    d = ransac.estimator_.intercept_  # intercept
    return a, b, d  # Z = aX + bY + d


if __name__ == '__main__':
    sample_pcd = o3d.io.read_point_cloud("./pc_color_chair.pcd")
    print(find_plane(sample_pcd))