import open3d as o3d
import numpy as np

def visualize_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.1500,#0.3412,
                                  front= [-0.4257, 0.2125, -0.7000], #[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])    
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.1500,#0.3412,
                                  front= [-0.4257, 0.2125, -0.7000], #[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    

# Mesh generation (compare 3 methods)
# http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html

# Ball pivoting algorithm
# Help from: 
#https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices


def BPA(cloud, radius):
    # estimate radius for rolling ball
    distances = cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius*avg_dist  
    
    radii = o3d.utility.DoubleVector([radius, radius * 2])
    mesh_BPA = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
    return mesh_BPA

def Poisson(cloud, depth=8, width=0, scale=1.1,linear_fit=False):
    mesh_poisson = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=depth, width=width, scale=scale, 
                                                                         linear_fit=linear_fit)[0]
    return mesh_poisson

def visualize_mesh(mesh):
    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " + str(mesh.has_vertex_colors()) + ")")
    o3d.visualization.draw_geometries([mesh],
                                  zoom=0.1500,#0.3412,
                                  front= [-0.4257, 0.2125, -0.7000], #[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])    
