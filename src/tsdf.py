
import open3d as o3d
import numpy as np
from read_write_model import read_images_text

#%%

def visualize_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.1500,
                                  front= [-0.4257, 0.2125, -0.7000],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])    
    
# Read trajectory from .log file 
# (functions from http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html#TSDF-volume-integration) 
class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

#%% Visualize the point cloud

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("../../3D-data/sfm_fused.ply")
#pcd = o3d.io.read_point_cloud('meshed-poisson.ply')
#print(pcd)
#print(np.asarray(pcd.points))
visualize_cloud(pcd)

#%%

# This function is based on this tutorial:
# https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
	
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
	
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
	
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
	
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
						   
    return rot_matrix

#%%

images = read_images_text("../../3D-data/images.txt")

#%%

quaternion = []
translation = []
for i in range(len(images)):
    quaternion.append(images[i+1][1])
    translation.append(images[i+1][2])

quaternion = np.array(quaternion)
translation = np.array(translation)

print(quaternion.shape)
print(translation.shape)

#%%

transformation_matrix = []
for i in range(len(quaternion)):
    rotation_matrix = quaternion_rotation_matrix(quaternion[i])
    transformation = np.c_[rotation_matrix, translation[i]]
    transformation = np.vstack((transformation, np.array([0,0,0,1])))
    transformation_matrix.append(transformation)

#%% Try to visualize depthmaps from COLMAP, using their scripts/python/read_dense.py

#from read_write_dense import read_array

#read_array('../../3D-data/depth_maps/')

#%% Follow this tutorial
#http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html#TSDF-volume-integration

# TODO: Find out if we have this or how this is generated
camera_poses = read_trajectory("../../3D-data/odometry.log") # Use example from tutorial

#%% TSDF volume integration
#TODO: Find out how to access the depthmaps
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("../../3D-data/images/{:05d}.jpg".format(i))
    depth = o3d.io.read_image("../../3D-data/depth_maps/{:05d}.png".format(i))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))


#%% Extract a mesh
print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
                                  front=[0.5297, -0.1873, -0.8272],
                                  lookat=[2.0712, 2.0312, 1.7251],
                                  up=[-0.0558, -0.9809, 0.1864],
                                  zoom=0.47)





#%%
def main():
    print()

if __name__ == '__main__':
    main()























