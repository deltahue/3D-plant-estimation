import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils import visualize_mesh

# Generates a poisson reconstructed mesh for a given pointcloud pcd
def generate_mesh(pcd, visualize = False):

    
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10)
    print(poisson_mesh)
    
    '''
    if visualize == True:
        o3d.visualization.draw_geometries([poisson_mesh])

        print('visualize densities')
    '''
    print("length: " + str(len(np.asarray(poisson_mesh.vertices))))
    
    
    # Make a density map and visualize it
    densities = np.asarray(densities)
    print(np.min(densities))
    print(np.max(densities))
    print(np.mean(densities))
    print(np.quantile(densities,0.25))
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = poisson_mesh.vertices
    density_mesh.triangles = poisson_mesh.triangles
    density_mesh.triangle_normals = poisson_mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    
    #if visualize == True:
    #    visualize_mesh(density_mesh)

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.25) #max(3.2, np.quantile(densities, 0.25))
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
    print(poisson_mesh)
    print("length after removal: " + str(len(np.asarray(poisson_mesh.vertices))))
    
    bbox = pcd.get_axis_aligned_bounding_box()
    poisson_mesh = poisson_mesh.crop(bbox)  
    print("length after bounding box: " + str(len(np.asarray(poisson_mesh.vertices))))

    if visualize == True:
        o3d.visualization.draw_geometries([poisson_mesh])    
    
    return poisson_mesh


# Takes in a (Poisson reconstructed) mesh and returns a Taubin smoothed mesh with num_iter iterations
def smooth_mesh(mesh, num_iter = 10, visualize = False):
    print('filter with Taubin with '+ str(num_iter) + ' iterations')
    mesh_taub = mesh.filter_smooth_taubin(num_iter, 0.5, -0.53)
    mesh_taub.compute_vertex_normals()
    
    if visualize == True:
        o3d.visualization.draw_geometries([mesh_taub])

    return mesh_taub

def remove_islands(mesh, visualize = True):
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    mesh_0 = mesh
    # Leave only the largest cluster in the mesh
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < np.max(cluster_n_triangles)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    
    if visualize == True:
        o3d.visualization.draw_geometries([mesh_0])

    return mesh_0

def remove_infs_nans(mesh):
    '''
    Remove all nans and infs from a generated mesh
    '''
    to_delete = []
    vertices = np.asarray(mesh.vertices)
    for i in range(len(vertices)):
        if np.isnan(vertices[i,0]) == True or np.isinf(vertices[i,0] == True):
            to_delete.append(i)
    mesh.remove_vertices_by_index(to_delete)
    
    return mesh



