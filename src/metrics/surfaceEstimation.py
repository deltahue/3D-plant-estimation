import numpy as np
import open3d as o3d
from numpy.linalg import norm
import trimesh.graph as trimesh

def findAreaOfTop(meshPath):
	print("Testing mesh in open3d ...")
	#mesh = o3d.io.read_triangle_mesh("./leafPointCloud.ply")
	#mesh = o3d.io.read_triangle_mesh("./leaf.obj")
	mesh = o3d.io.read_triangle_mesh(meshPath)
	#mesh = o3d.io.read_triangle_mesh("./bottle.obj")
	tris = np.asarray(mesh.triangles)
	vers = np.asarray(mesh.vertices)
	mesh.compute_vertex_normals(normalized=True)
	mesh.compute_triangle_normals(normalized=True)
	#print(mesh.has_triangle_normals())
	#print(mesh.has_vertex_normals())
	tri_normals = np.asarray(mesh.triangle_normals)
	#print(tri_normals)
	
	#print(tris)
	#print(vers)
	#print("nors: ", nors)
	#print("here")
	#sum = 0
	#i = 0
	#while(True):
	#	a = mesh.triangles
	#for i in range(len(mesh.triangles)):
	#	i += 3
	#	Vector3 corner = vertices[triangles[i]];
        #Vector3 a = vertices[triangles[i + 1]] - corner;
        #Vector3 b = vertices[triangles[i + 2]] - corner;

        #sum += Vector3.Cross(a, b).magnitude;
	summ = 0
	#print("len: ", len(tris))
	for i in range(len(tris)):
		#print("i: ", i)
		tri = tris[i]
        #print(tri)
		#print("tri:", tri)
		#print("ele: ", tri[0])
		#print(vers[tri[0]])
		v1 = vers[tri[0]]
		v2 = vers[tri[1]]
		v3 = vers[tri[2]]
        #print("v1: ", v1)
		area = np.cross(v2 - v1, v3 - v1) / 2
        #print("area1: ", area)
		area = norm(area, 2)
		#print("area2: ", area)
		summ += area
	#print("sum: ", summ)
	listAdj = trimesh.face_adjacency(faces=tris)
	#print("adjacent faces: ", listAdj)
	adjs = []
	for i in range(len(tris)):
		adjs.append([])
	#print(adjs)
	for pair in listAdj:	
		#print(pair)
		#print(pair[0])
		adjs[pair[0]].append(pair[1])
		adjs[pair[1]].append(pair[0])
	#print(adjs)
	
	#find one triangal facing up
	#for(face in tris):
	#	v1 = vers[face[0]]
	#	v2 = vers[face[1]]
        #	v3 = vers[face[2]]
        #	#print("v1: ", v1)
        #	area = np.cross(v2 - v1, v3 - v1)
	#	#norTri = 
	#o3d.visualization.draw_geometries([mesh])
	
	#pcd = o3d.geometry.PointCloud()
	#pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
	#pcd.estimate_normals()
	#o3d.visualization.draw_geometries([pcd])
	
	return summ



"""
if __name__ == "__main__":
	findAreaOfTop((1,0,0)) 	
"""	



