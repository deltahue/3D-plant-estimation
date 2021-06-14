import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Generates a poisson reconstructed mesh for a given pointcloud pcd
def generate_mesh(pcd, visualize = False):
    '''
    Returns a Poisson reconstructed mesh with the lowest densities 
    and parts of outside the original point cloud removed
    '''
    
    # Use depth of 10. Get the densities so we can remove the lowest densities later
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10)
    
    # Make a density map
    densities = np.asarray(densities)

    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.25) 
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
    #print(poisson_mesh)
    #print("length after removal: " + str(len(np.asarray(poisson_mesh.vertices))))
    
    # Remove any parts of the mesh that are outside the original bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    poisson_mesh = poisson_mesh.crop(bbox)  
    #print("length after bounding box: " + str(len(np.asarray(poisson_mesh.vertices))))

    if visualize == True:
        o3d.visualization.draw_geometries([poisson_mesh])    
    
    return poisson_mesh

def smooth_mesh(mesh, num_iter = 10, visualize = False):
    '''
    Returns a Taubin smoothed mesh with num_iter iterations
    '''
    #print('filter with Taubin with '+ str(num_iter) + ' iterations')
    mesh_taub = mesh.filter_smooth_taubin(num_iter, 0.5, -0.53)
    mesh_taub.compute_vertex_normals()
    
    if visualize == True:
        o3d.visualization.draw_geometries([mesh_taub])

    return mesh_taub

def remove_islands(mesh, visualize = True):
    '''
    Remove only the largest part of the mesh 
    (removes small islands caused by remaining noise)
    '''
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



