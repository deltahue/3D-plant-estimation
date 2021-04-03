
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
#mesh = o3d.io.read_triangle_mesh('mesh_poi_stat_cl_30_1p5_dep12.ply')
#print(pcd)
#print(np.asarray(pcd.points))
print(mesh)
#visualize_cloud(mesh)

#%%
# Some inspiration from
# https://stackoverflow.com/questions/61269980/open3d-crop-pointcloud-with-polygon-volume
'''
corners = np.array([[ -9, -9, 25 ],
		[ -9, 10, 25 ],
		[ 19, -9, 25 ],
		[ 19, 10, 25],
		[ -9, -9, 1 ],
		[ -9, 10, 1 ],
		[ 19, -9, 1 ],
		[ 19, 10, 1 ]])
'''

corners = np.array([
        [ -5, -5, 20 ],
		[ -5, 10, 20 ],
		[ 10, -5, 20 ],
		[ 10, 10, 20],
		[ -5, -5, 1 ],
		[ -5, 10, 1 ],
		[ 10, -5, 1 ],
		[ 10, 10, 1 ]])

		
# Convert the corners array to have type float64
bounding_polygon = corners.astype("float64")

# Create a SelectionPolygonVolume
vol = o3d.visualization.SelectionPolygonVolume()

# Specify what axis to orient the polygon to.
# Orient the polygon to the "Y" axis. Max value the maximum Y of
# the polygon vertices and the min value the minimum Y of the polygon vertices.
vol.orthogonal_axis = "Y"
vol.axis_max = np.max(bounding_polygon[:, 1])
vol.axis_min = np.min(bounding_polygon[:, 1])

# Set all the Y values to 0 (they aren't needed since we specified what they
# should be using just vol.axis_max and vol.axis_min).
bounding_polygon[:, 1] = 0

# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

# Crop the point cloud using the Vector3dVector
cropped_pcd = vol.crop_point_cloud(pcd)

# Get a nice looking bounding box to display around the newly cropped point cloud
# (This part is optional and just for display purposes)
bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)

# Draw the newly cropped PCD and bounding box
#visualize_cloud(cropped_pcd)

o3d.visualization.draw_geometries([cropped_pcd, bounding_box],
                                  zoom=1,#0.3412,
                                  front= [-0.4257, 0.2125, -0.7000], #[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])  


'''
vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
plant = vol.crop_point_cloud(pcd)
print(plant)
visualize_cloud(plant)
'''

#%%

print("Make a partial mesh")


meshvertices = np.asarray(mesh.vertices)
meshvertices_cropped = np.where(np.asarray(mesh.vertices)>15, np.asarray(mesh.vertices), -1)
meshtriang = np.asarray(mesh.triangles)
to_delete = []
for i in range(len(meshvertices_cropped)):
    if meshvertices_cropped[i][0] == -1 or meshvertices_cropped[i][2] == -1 or meshvertices_cropped[i][2] == -1:
        to_delete.append(i)
meshvertices_cropped = np.delete(meshvertices_cropped, to_delete,0)
#meshtriang = np.delete(meshtriang, to_delete,0)

print("make the mesh")
mesh.vertices = o3d.utility.Vector3iVector(meshvertices_cropped)

o3d.visualization.draw_geometries([mesh])


'''
mesh1 = mesh
meshtriang = np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2 :]
meshtriangnorm = np.asarray(mesh1.triangle_normals)[:len(mesh1.triangles) // 2 :]

print("We make a partial mesh of only the first half triangles.")
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2 :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])
'''

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

#%% # Set up experiments: (let several experiments run and visualize later)

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

#%% Old visualizations

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












