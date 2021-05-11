import numpy as np
import open3d as o3d
from numpy import linalg as LA
from numpy.linalg import norm
import trimesh.graph as trimesh
import math


def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds)

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
        cone_height=cone_height,
        cylinder_radius=0.5,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(-1 *Ry, center=np.array([0, 0, 0]))
    mesh.rotate(-1 *Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)
    
 



def findNormalPCA(croppedMesh):
	mesh = o3d.io.read_triangle_mesh(croppedMesh)
	print("here")
	tris = np.asarray(mesh.triangles)
	P = np.asarray(mesh.vertices)
	print(P)
	mean = P.mean(axis=0)
	print(mean)
	print(P-mean)
	cov = np.matmul(np.transpose(P-mean), (P-mean))
	print("cov: ", cov)
	w, v = LA.eig(np.array(cov))
	print("w: ", w)
	print("v: ", v)
	return v

def getHeight(croppedMesh, normal, pointOnFloor):
    mesh = o3d.io.read_triangle_mesh(croppedMesh)
    print("here")
    P = np.asarray(mesh.vertices)
    maxHeight = 0
    for p in P:
        vec = p - pointOnFloor
        projVec = np.dot(vec, normal)
        print("projVec: ", projVec)
        #print("vec: ", vec)
        if(projVec > maxHeight):
            maxHeight = projVec 
            maxVec = vec
    origin = pointOnFloor
    maxHeight /= math.sqrt(np.dot(normal, normal))
    drawNormal(croppedMesh,  (normal/math.sqrt(np.dot(normal, normal))) * maxHeight, origin)
    return maxHeight

def drawNormal(meshPath, normal,origin):
    #norVec = get_arrow(origin=[0,0,0], vec=normal)
    #nor2Vec = get_arrow(origin=[0,1,0], vec=nor)
    norVec = get_arrow(origin=origin, vec=-1*normal)
    #nor2Vec = get_arrow(origin=origin, vec=-1*nor)
    mesh = o3d.io.read_triangle_mesh(meshPath)
    o3d.visualization.draw_geometries([norVec, mesh])

    
"""    
if __name__=="__main__":
	mesh = o3d.io.read_triangle_mesh("./cropped_mesh_colmap(plant).ply")
	v = findNormalPCA()
	
	# Create a Cartesian Frame of Reference
	FOR = get_o3d_FOR()

	arrow1 = get_arrow(origin=[0,0,0], vec=v[0]*10)
	arrow2 = get_arrow(origin=[0,0,0], vec=v[1]*10)
	arrow3 = get_arrow(origin=[0,0,0], vec=v[2]*10)
	
	# Draw everything
	o3d.visualization.draw_geometries([arrow1, arrow2, arrow3, mesh])
	
	april = ap.April()
	normal = april.findNormal()
	norVec = get_arrow(origin=[0,0,0], vec=normal*10)
	mesh = o3d.io.read_triangle_mesh("./bigPlantDataSetWithAprilTag_lowres/dense/0/meshed-poisson.ply")
	o3d.visualization.draw_geometries([norVec,mesh])
	
"""
