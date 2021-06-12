import math
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def findAngle(leafPath, normal):
    #print("hello")
    mesh = o3d.io.read_triangle_mesh(leafPath)
    mesh2 = o3d.io.read_point_cloud(leafPath)
    #print("here")
    P = np.asarray(mesh.vertices)
    #print("P: ", P)
    tris = np.asarray(mesh.triangles)
    arr = []
    i = 0
    mesh.compute_triangle_normals(normalized=True)
    mesh2.orient_normals_to_align_with_direction()
    #mesh2.orient_normals_towards_camera_location()
    #o3d.visualization.draw_geometries([mesh2])
    
    #tri_normals = np.asarray(mesh2.normals)
    tri_normals = np.asarray(mesh.triangle_normals)
    #print("normals: ", tri_normals)
    normal = -1 * normal
    for tri in tri_normals:
        #if i == 200:
        #    break
        #print(mesh.has_triangle_normals())
        #print(mesh.has_vertex_normals())
        #A.B = |A|x|B|x cos(X)
        vecNor = tri_normals[i]
        cosAng = np.dot(vecNor, normal)/math.sqrt((np.dot(vecNor, vecNor) * np.dot(normal, normal)))
        #print("cos angle: ", cosAng)
        i += 1
        ang = math.degrees(math.acos(cosAng))
        if(ang > 90):
            ang = 180 - ang
        arr.append(ang)

    #print("arr: ", arr)
    #print("length: ", len(arr))
    
    #plt.hist(np.array(arr), bins=10)

    #np.random.seed(42)
    #x = np.random.normal(size=1000)

    return arr
        
