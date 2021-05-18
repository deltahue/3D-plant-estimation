# Experiment with filling mesh holes

import open3d as o3d
import numpy as np
#import trimesh as tri


mesh = o3d.io.read_triangle_mesh("/home/soley/3D-data/meshes_avocado/mesh_label5.ply")
o3d.visualization.draw_geometries([mesh])
#%%
print("Cluster connected triangles")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print("Show mesh with small clusters removed")
mesh_0 = mesh
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh_0.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_0])
