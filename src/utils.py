'''
This file contains basic utility functions, 
for computation, visualization and saving
'''

import open3d as o3d
import numpy as np


# Compute rotation matrix that rotates rot_x degrees around the x-axis,
# rot_y degrees around the y-axis and rot_z degrees around the z-axis
def compute_rotation_matrix(rot_x, rot_y, rot_z):
    Rx = np.array([[1,0,0],
                   [0,np.cos(np.radians(rot_x)),np.sin(np.radians(rot_x))*-1],
                   [0,np.sin(np.radians(rot_x)),np.cos(np.radians(rot_x))]]),
    Ry= np.array([[np.cos(np.radians(rot_y)),0,np.sin(np.radians(rot_y))],
                   [0,1,0],
                   [-1*np.sin(np.radians(rot_y)),0,np.cos(np.radians(rot_y))]]),
    Rz = np.array([[np.cos(np.radians(rot_z)),-1*np.sin(np.radians(rot_z)),0],
                   [np.sin(np.radians(rot_z)),np.cos(np.radians(rot_z)),0],
                   [0,0,1]]),
    
    return np.resize(np.matmul(Rz, np.matmul(Ry, Rx)), (3,3))


# Visualize point cloud
def visualize_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.1500,
                                  front= [-0.4257, 0.2125, -0.7000],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])    
    
# Visualize point cloud outliers
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.1500,#0.3412,
                                  front= [-0.4257, 0.2125, -0.7000], 
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])    
    
    
def create_bounding_box(cen, ext, rotations):
    cen = np.array([1.5,1,2])
    ext = np.array([5,5,4])
    rot_matrix = compute_rotation_matrix(rotations[0],rotations[1],rotations[2])
    
    bbox = o3d.geometry.OrientedBoundingBox(center = cen,
                                            R = rot_matrix,
                                            extent = ext
                                            )
    
    bbox.color= (0,1,0)
    
    minb = np.array([cen[0] - ext[0]/2,cen[1] - ext[1]/2,cen[2] - ext[2]/2])
    maxb = np.array([cen[0] + ext[0]/2,cen[1] + ext[1]/2,cen[2] + ext[2]/2])
    aabbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = minb, max_bound = maxb)
    aabbox.color= (1,0,0)
    
    return bbox, aabbox
    
# Saves point cloud to the path (string)
def save_point_cloud(path, pcd):
    o3d.io.write_point_cloud(path, pcd)
    
# Saves mesh to the path (string)
def save_mesh(path, mesh):
    o3d.io.write_point_cloud(path, mesh)
        
        
        
        
        

