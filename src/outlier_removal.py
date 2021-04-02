
import open3d as o3d
import numpy as np
import time

from utils import visualize_cloud, visualize_mesh, display_inlier_outlier, BPA, Poisson

#TODO: Before exporting, downsample to acceptable number of triangles

#%% # Start following this link to visualize the point cloud
#http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

print("Load a ply point cloud, print it, and render it")
#pcd = o3d.io.read_point_cloud("big_with_apriltags_midres_fused.ply")
pcd = o3d.io.read_point_cloud("../../3D-data/sfm_fused.ply")
mesh = o3d.io.read_triangle_mesh('../../3D-data/meshed-poisson.ply')
#print(pcd)
#print(np.asarray(pcd.points))
print(mesh)
visualize_cloud(mesh)
#%% Downsample the point cloud
# (from here on, we are following http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)

print("Downsample the point cloud with a voxel of 0.02")
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
visualize_cloud(voxel_down_pcd)


#%% statistical outlier removal

print("Statistical outlier removal")
stat_cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5) # was 20, 2.0 
print("number of outliers is: " + str(len(pcd.compute_nearest_neighbor_distance()) - len(ind)))
display_inlier_outlier(voxel_down_pcd, ind)
visualize_cloud(stat_cl)

#%% Radius outlier removal

print("Radius oulier removal")
rad_cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.1) # Was 16, 0.05
display_inlier_outlier(voxel_down_pcd, ind)
visualize_cloud(rad_cl)

#%% # Set up experiments: (experiment first, then let it run and visualize later)

print(time.asctime())
print("Statistical outlier removal")
stat_cl_30_1p5, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

mesh_BPA_stat_cl_30_1p5_rad1p5 =  BPA(stat_cl_30_1p5, 1.5)
mesh_poi_stat_cl_30_1p5_default = Poisson(stat_cl_30_1p5)
print(time.asctime())
mesh_BPA_stat_cl_30_1p5_rad2p0 =  BPA(stat_cl_30_1p5, 2.0)
print(time.asctime())
mesh_BPA_stat_cl_30_1p5_rad2p5 =  BPA(stat_cl_30_1p5, 2.5)
print(time.asctime())
mesh_BPA_stat_cl_30_1p5_rad3p0 =  BPA(stat_cl_30_1p5, 3.0)
print(time.asctime())
mesh_BPA_stat_cl_30_1p5_rad4p0 =  BPA(stat_cl_30_1p5, 4.0)
print(time.asctime())
mesh_BPA_stat_cl_30_1p5_rad5p0 =  BPA(stat_cl_30_1p5, 5.0)

mesh_poi_stat_cl_30_1p5_dep5 = Poisson(stat_cl_30_1p5, depth = 5)
mesh_poi_stat_cl_30_1p5_dep7 = Poisson(stat_cl_30_1p5, depth = 7)
mesh_poi_stat_cl_30_1p5_scale1p5 = Poisson(stat_cl_30_1p5, scale = 1.5)
mesh_poi_stat_cl_30_1p5_scale2p0 = Poisson(stat_cl_30_1p5, scale = 2.0)
mesh_poi_stat_cl_30_1p5_dep9 = Poisson(stat_cl_30_1p5, depth = 9)
mesh_poi_stat_cl_30_1p5_dep10 = Poisson(stat_cl_30_1p5, depth = 10) 

mesh_poi_stat_cl_30_1p5 = Poisson(stat_cl_30_1p5)

print(time.asctime())
print("Radius outlier removal")
rad_cl_16_0p1, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius = 0.1)

mesh_poi_rad_cl_16_0p1 = Poisson(stat_cl_30_1p5)

mesh_BPA_rad_cl_16_0p1_rad1p5 =  BPA(rad_cl_16_0p1, 1.5)
print(time.asctime())
mesh_poi_rad_cl_16_0p1_default = Poisson(rad_cl_16_0p1) 
mesh_BPA_rad_cl_16_0p1_rad2p0 =  BPA(rad_cl_16_0p1, 2.0)
print(time.asctime())
mesh_BPA_rad_cl_16_0p1_rad2p5 =  BPA(rad_cl_16_0p1, 2.5)
print(time.asctime())
mesh_BPA_rad_cl_16_0p1_rad3p0 =  BPA(rad_cl_16_0p1, 3.0) 
print(time.asctime())
mesh_BPA_rad_cl_16_0p1_rad4p0 =  BPA(rad_cl_16_0p1, 4.0) 
print(time.asctime())
mesh_BPA_rad_cl_16_0p1_rad5p0 =  BPA(rad_cl_16_0p1, 5.0)

mesh_poi_rad_cl_16_0p1_dep5 = Poisson(rad_cl_16_0p1, depth = 5)
mesh_poi_rad_cl_16_0p1_dep7 = Poisson(rad_cl_16_0p1, depth = 7)
mesh_poi_rad_cl_16_0p1_scale1p5 = Poisson(rad_cl_16_0p1, scale = 1.5)
mesh_poi_rad_cl_16_0p1_scale2p0 = Poisson(rad_cl_16_0p1, scale = 2.0)


print(time.asctime()) 

#%% Visualize the results

stat_cl_30_1p5, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
mesh_poi_stat_cl_30_1p5_dep12 = Poisson(stat_cl_30_1p5, depth = 12) 
visualize_mesh(mesh_poi_stat_cl_30_1p5_dep12)

'''
my_mesh = Poisson(pcd) # Basic, no outlier removal
visualize_mesh(my_mesh)

visualize_mesh(mesh_poi_stat_cl_30_1p5)
visualize_mesh(mesh_poi_rad_cl_16_0p1)

visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad1p5)
visualize_mesh(mesh_poi_stat_cl_30_1p5_default)
visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad2p0)
visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad2p5)
visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad3p0)
visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad4p0)
visualize_mesh(mesh_BPA_stat_cl_30_1p5_rad5p0)

visualize_mesh(mesh_poi_stat_cl_30_1p5_dep5)
visualize_mesh(mesh_poi_stat_cl_30_1p5_dep7)

visualize_mesh(mesh_poi_stat_cl_30_1p5_scale1p5)
visualize_mesh(mesh_poi_stat_cl_30_1p5_scale2p0)

visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad1p5)
visualize_mesh(mesh_poi_rad_cl_16_0p1_default)

visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad2p0)
visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad2p5)
visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad3p0)
visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad4p0)
visualize_mesh(mesh_BPA_rad_cl_16_0p1_rad5p0)

visualize_mesh(mesh_poi_rad_cl_16_0p1_dep5)
visualize_mesh(mesh_poi_rad_cl_16_0p1_dep7)
visualize_mesh(mesh_poi_rad_cl_16_0p1_scale1p5)
visualize_mesh(mesh_poi_rad_cl_16_0p1_scale2p0)
visualize_mesh(mesh_poi_stat_cl_30_1p5_dep9)
visualize_mesh(mesh_poi_stat_cl_30_1p5_dep10)
'''



'''
print(".")
stat_cl_25_1p5, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)
mesh_BPA_stat_cl_25_1p5_rad1p5 =  BPA(stat_cl_25_1p5, 1.5)
mesh_poi_stat_cl_30_1p5_default = Poisson(stat_cl_30_1p5)
print(".")
stat_cl_30_2p0, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
#mesh_BPA_stat_cl_30_1p5_rad1p5 =  BPA(stat_cl_30_1p5, 1.5)
#mesh_poi_stat_cl_30_1p5_default = Poisson(stat_cl_30_1p5)
print(".")
stat_cl_25_2p0, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
#mesh_BPA_stat_cl_30_1p5_rad1p5 =  BPA(stat_cl_30_1p5, 1.5)
#mesh_poi_stat_cl_30_1p5_default = Poisson(stat_cl_30_1p5)
print(".")
'''












