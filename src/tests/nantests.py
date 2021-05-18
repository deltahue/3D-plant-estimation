import open3d as o3d
import numpy as np


for i in range(7):
    mesh = o3d.io.read_triangle_mesh("/home/soley/3D-data/meshes_luca2/mesh_label"+ str(i) + ".ply")

    flag = 0
    vertices = np.asarray(mesh.vertices)
    print(i)
    
    if np.max(np.isnan(vertices)) == True:
        flag += 1
    print(np.max(np.isnan(vertices)))
    print(np.max(np.isinf(vertices)))
    print()
    # Nan in 1, 8 and 9 of old meshes, zero in current
    
if flag == 0:
    print("No nans")
else:
    print("Some of the meshes contain nans")
#%% Take a closer look at specific meshes if they contain nan

mesh = o3d.io.read_triangle_mesh("/home/soley/3D-data/old_meshes/mesh_label1.ply")

vertices = np.asarray(mesh.vertices)
print(vertices)

for i in range(vertices.shape[0]):
    for j in range(vertices.shape[1]):
        if np.isnan(vertices[i][j]) == True :
            print(str(i) + "," + str(j))
            