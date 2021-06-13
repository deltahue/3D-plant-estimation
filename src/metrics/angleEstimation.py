import math
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def findAngle(leafPath, normal):
    mesh = o3d.io.read_triangle_mesh(leafPath)
    mesh2 = o3d.io.read_point_cloud(leafPath)
    P = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    arr = []
    i = 0
    mesh.compute_triangle_normals(normalized=True)
    mesh2.orient_normals_to_align_with_direction()
    
    tri_normals = np.asarray(mesh.triangle_normals)
    normal = -1 * normal
    for tri in tri_normals:

        #A.B = |A|x|B|x cos(X)
        vecNor = tri_normals[i]
        cosAng = np.dot(vecNor, normal)/math.sqrt((np.dot(vecNor, vecNor) * np.dot(normal, normal)))
        #print("cos angle: ", cosAng)
        i += 1
        ang = math.degrees(math.acos(cosAng))
        if(ang > 90):
            ang = 180 - ang
        arr.append(ang)

    return arr
        
