
import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import visualize_cloud, visualize_mesh, display_inlier_outlier, Poisson, compute_rotation_matrix

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

#%% Bounding box to crop the mesh to just the plant

# Need to rotate the bounding box 45 degrees to have the plant approximately in center
bbox_whole = o3d.geometry.OrientedBoundingBox(center = np.array([2.5,-2,4.5]),
                                        R = compute_rotation_matrix(45,0,0),
                                        extent = np.array([5,5,4.5])
                                        )

#%% Bounding box to crop individual leaves

#rot_x, rot_y, rot_z = 20,-40,55
rot_x, rot_y, rot_z = 0,40,0
rot_matrix = compute_rotation_matrix(rot_x, rot_y, rot_z)
#cen = np.array([2,2.5,6.5])
#ext = np.array([20,10,19])
cen = np.array([2.2,0.1,4.5])
ext = np.array([1.2,1,1.4]) # was x=1.5
 

bbox = o3d.geometry.OrientedBoundingBox(center = cen,
                                        R = rot_matrix,
                                        extent = ext
                                        )
bbox.color= (0,1,0)

minb = np.array([cen[0] - ext[0]/2,cen[1] - ext[1]/2,cen[2] - ext[2]/2])
maxb = np.array([cen[0] + ext[0]/2,cen[1] + ext[1]/2,cen[2] + ext[2]/2])
aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound = minb, max_bound = maxb)
aabb.color= (1,0,0)


#%% Crop the point cloud and export results

cropped_pcd = pcd.crop(bbox)
visualize_cloud(cropped_pcd)
#o3d.visualization.draw_geometries([cropped_pcd, bbox,aabb])
#o3d.io.write_point_cloud('../../3D-data/cropped_pcd_raw_smaller.ply', cropped_pcd)
#o3d.io.write_point_cloud('../../3D-data/leaf_raw_pointcloud.ply', cropped_pcd)


#%% Crop the mesh
'''
mesh1 = o3d.io.read_triangle_mesh('../../3D-data/meshed-poisson.ply')
print(mesh1)

cropped_mesh = mesh1.crop(bbox)
print(cropped_mesh)
o3d.visualization.draw_geometries([cropped_mesh, bbox,aabb])
#visualize_cloud(cropped_mesh)
'''

#%%
# Export results
#o3d.io.write_triangle_mesh('../../3D-data/cropped_mesh_smaller.ply', cropped_mesh)
#%% Downsample and remove outliers from cropped point cloud

print("Downsample the point cloud with a voxel of 0.02")
voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
#visualize_cloud(voxel_down_cropped_pcd)


print("Radius oulier removal")
rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=80, radius=0.1)
print("number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
display_inlier_outlier(voxel_down_cropped_pcd, ind)
visualize_cloud(rad_cl)

#%%

#o3d.io.write_point_cloud('../../3D-data/cropped_pcd_leaf_filtered_rad80_0p1.ply', rad_cl)

#%% same for whole point cloud

#cropped_pcd = pcd.crop(bbox_whole)
cropped_pcd = pcd
visualize_cloud(cropped_pcd)

voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
#visualize_cloud(voxel_down_cropped_pcd)

print("Radius oulier removal")
rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=10, radius=0.1)
print("number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
display_inlier_outlier(voxel_down_cropped_pcd, ind)
visualize_cloud(rad_cl)

#%% Finally, try mesh generation on the filtered pointcloud
print(time.asctime())
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        rad_cl, depth=9)
print(poisson_mesh)
print(time.asctime())
#%% 
visualize_mesh(poisson_mesh)

#%%

print('visualize densities')
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = poisson_mesh.vertices
density_mesh.triangles = poisson_mesh.triangles
density_mesh.triangle_normals = poisson_mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
visualize_mesh(density_mesh)

#%%

print('remove low density vertices')
vertices_to_remove = densities < np.quantile(densities, 0.1)
poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
print(poisson_mesh)
visualize_mesh(poisson_mesh)

#%%
#o3d.io.write_triangle_mesh('../../3D-data/own_mesh.ply', poisson_mesh)
#o3d.io.write_triangle_mesh('../../3D-data/leaf_mesh.ply', poisson_mesh)


#%% statistical outlier removal
'''
print("Statistical outlier removal")
stat_cl, ind = voxel_down_cropped_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5) # was 20, 2.0 
print("number of outliers is: " + str(len(pcd.compute_nearest_neighbor_distance()) - len(ind)))
display_inlier_outlier(voxel_down_cropped_pcd, ind)
visualize_cloud(stat_cl)
'''




