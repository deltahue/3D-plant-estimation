import numpy as np
import open3d as o3d
from numpy.linalg import norm
import trimesh.graph as trimesh

def findAreaOfTop(meshPath):
	mesh = o3d.io.read_triangle_mesh(meshPath)
	#sumNew = mesh.get_surface_area()
	tris = np.asarray(mesh.triangles)
	vers = np.asarray(mesh.vertices)
	mesh.compute_vertex_normals(normalized=True)
	mesh.compute_triangle_normals(normalized=True)

	tri_normals = np.asarray(mesh.triangle_normals)
	summ = 0
	for i in range(len(tris)):
		tri = tris[i]
		v1 = vers[tri[0]]
		v2 = vers[tri[1]]
		v3 = vers[tri[2]]
		area = np.cross(v2 - v1, v3 - v1) / 2
		area = norm(area, 2)
		summ += area
	listAdj = trimesh.face_adjacency(faces=tris)
	adjs = []
	for i in range(len(tris)):
		adjs.append([])
	for pair in listAdj:	
		adjs[pair[0]].append(pair[1])
		adjs[pair[1]].append(pair[0])
	
	return summ
