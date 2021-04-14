
import open3d as o3d
import numpy as np
import time

from utils import visualize_cloud, visualize_mesh, display_inlier_outlier, BPA, Poisson

#TODO: Before exporting, downsample to acceptable number of triangles

#%% # Start following this link to visualize the point cloud
#http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("../../3D-data/sfm_fused.ply")
mesh = o3d.io.read_triangle_mesh('../../3D-data/meshed-poisson.ply')
#mesh = o3d.io.read_triangle_mesh('../../3D-data/mesh_poi_stat_cl_30_1p5_dep12.ply')
#print(pcd)
#print(np.asarray(pcd.points))
print(mesh)
#visualize_cloud(pcd)
visualize_cloud(mesh)

#%%
# Try to crop mesh
# https://github.com/intel-isl/Open3D/issues/1410

mesh1 = o3d.io.read_triangle_mesh('../../3D-data/meshed-poisson.ply')
print(mesh1)

# Need to rotate the bounding box 45 degrees to have the plant approximately in center
bbox = o3d.geometry.OrientedBoundingBox(#center = np.array([2.5,-0.25,4.5]),
                                        center = np.array([2.5,-2,4.5]),
                                        #R = np.array([[1,0,0],[0,1,0],[0,0,1]]),
                                        #R = np.array([[1,0,0],[0,3/np.sqrt(2),-1/2],[0,1/2,3/np.sqrt(2)]]),
                                        #R = np.array([[1,0,0],[0,1/np.sqrt(2),-1/np.sqrt(2)],[0,1/np.sqrt(2),1/np.sqrt(2)]]),
                                        R = np.array([[1,0,0],[0,np.cos(np.radians(45)),-1* np.sin(np.radians(45))],[0,np.cos(np.radians(45)),np.sin(np.radians(45))]]),
                                        #R = np.matmul(
                                        #    np.array([[1,0,0],[0,1/np.sqrt(2),-1/np.sqrt(2)],[0,1/np.sqrt(2),1/np.sqrt(2)]]),
                                        #    np.array([[1/np.sqrt(2),0,1/np.sqrt(2)],[0,1,0],[-1/np.sqrt(2),0,1/np.sqrt(2)]])),
                                        #extent = np.array([5,7.5,5])
                                        extent = np.array([5,5,4.5])
                                        )

cropped_mesh = mesh1.crop(bbox)
print(cropped_mesh)
o3d.visualization.draw_geometries([cropped_mesh,bbox])
visualize_cloud(cropped_mesh)

#%%
# Export results
o3d.io.write_triangle_mesh('../../3D-data/cropped_mesh_smaller.ply', cropped_mesh)

#%% Also crop the point cloud and export results

cropped_pcd = pcd.crop(bbox)
visualize_cloud(cropped_pcd)
o3d.io.write_point_cloud('../../3D-data/cropped_pcd_raw_smaller.ply', cropped_pcd)

#%% Downsample and remove outliers from cropped point cloud


print("Downsample the point cloud with a voxel of 0.02")
voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
#visualize_cloud(voxel_down_cropped_pcd)

'''
print("Statistical outlier removal")
stat_cl, ind = voxel_down_cropped_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
print("number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(stat_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
display_inlier_outlier(voxel_down_cropped_pcd, ind)
visualize_cloud(stat_cl)
'''

print("Radius oulier removal")
rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=20, radius=0.1)
print("number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
display_inlier_outlier(voxel_down_cropped_pcd, ind)
visualize_cloud(rad_cl)

#%%

o3d.io.write_point_cloud('../../3D-data/cropped_pcd_smaller_downsampled_and_filtered_rad20.ply', rad_cl)

#%% Finally, try mesh generation on the filtered pointcloud

print(time.asctime()) 
mesh_poi_rad14_scale2 = Poisson(rad_cl, depth = 14, scale=2.0) 
print(time.asctime())

#%% 
visualize_mesh(mesh_poi_rad14_scale2)
o3d.io.write_triangle_mesh('../../3D-data/mesh_from_cropped_pcd.ply', mesh_poi_rad14_scale2)

#%%

print(time.asctime()) 
mesh_bpa_cropped = BPA(rad_cl, 4.0)
print(time.asctime())

#%% 
visualize_mesh(mesh_bpa_cropped)
o3d.io.write_triangle_mesh('../../3D-data/bpa_mesh_from_cropped_pcd.ply', mesh_bpa_cropped)

#%%

print(time.asctime()) 
mesh_bpa = BPA(pcd, 4.0)
print(time.asctime())

#%%

visualize_mesh(mesh_bpa)
#o3d.io.write_triangle_mesh('../../3D-data/bpa_mesh_from_unfiltered_cropped_pcd.ply', mesh_bpa)
=======
>>>>>>> 526c02234c3b8bf2942269dc5dd749f4bc31222f

#%% Downsample the point cloud
# (from here on, we are following http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)

voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

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

#%% # Set up experiments: (let several experiments run and visualize later)
'''
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
'''
#%% Visualize results

print(time.asctime()) 
'''
stat_cl_30_1p5, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
mesh_poi_stat_cl_30_1p5_dep12 = Poisson(stat_cl_30_1p5, depth = 12) 
visualize_mesh(mesh_poi_stat_cl_30_1p5_dep12)
'''
stat_cl_30_1p5, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
mesh_poi_dep12 = Poisson(voxel_down_pcd, depth = 12) 
print(time.asctime()) 
visualize_mesh(mesh_poi_dep12)

#%% Export results

#o3d.io.write_triangle_mesh('mesh_poi_stat_cl_30_1p5_dep12_scale2.ply', mesh_poi_stat_cl_30_1p5_dep12_scale2)











