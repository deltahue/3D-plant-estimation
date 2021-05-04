import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils import visualize_mesh

# Generates a poisson reconstructed mesh for a given pointcloud pcd
def generate_mesh(pcd, taubin_filter = False):

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print(poisson_mesh)
    
    o3d.visualization.draw_geometries([poisson_mesh])

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

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
    print(poisson_mesh)


    visualize_mesh(poisson_mesh)

    if taubin_filter == True:
        #num_iter = [5,10,50,100]
        num_iter = [10]
        for i in num_iter:
            print('filter with Taubin with '+ str(i) + ' iterations')
            mesh_taub = poisson_mesh.filter_smooth_taubin(i, 0.5, -0.53)
            mesh_taub.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh_taub])
        
        return mesh_taub
    
    return poisson_mesh
    





