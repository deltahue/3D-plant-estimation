
# import packages
import open3d as o3d
import numpy as np
import cv2 as cv

# colmap
from read_write_model import read_images_text, read_cameras_text
from read_write_dense import read_array

# Useful functions
from utils import visualize_cloud#, quaternion_rotation_matrix
from scipy.spatial.transform import Rotation as R

visualize_pcd = False
visualize_mesh = True
save_depth_maps = True
    
# Visualize the point cloud
if visualize_pcd == True:
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("../../3D-data/sfm_fused.ply")
    visualize_cloud(pcd)


#%%

# Read information about camera poses
images = read_images_text("../../3D-data/images.txt")

# Convert information into (world coordinate) transformation matrices
quaternion = []
translation = []
for i in range(len(images)):
    quaternion.append(images[i+1][1])
    translation.append(images[i+1][2])

quaternion = np.array(quaternion)
translation = np.array(translation)

transformation_matrices = []
for i in range(len(quaternion)):
    rotation_matrix = R.from_quat(quaternion[i]).as_matrix()
    #rotation_matrix = quaternion_rotation_matrix(quaternion[i])
    transformation = np.c_[rotation_matrix, translation[i]]
    transformation = np.vstack((transformation, np.array([0,0,0,1])))
    transformation_matrices.append(transformation)

transformation_matrices = np.array(transformation_matrices)

# Convert the world coordinate matrices into trajectories
trajectories = []
trajectories.append(np.eye(4)) # First frame is the reference
for i in range((len(transformation_matrices)-1)):
    trans = np.matmul(transformation_matrices[i+1],np.linalg.inv(transformation_matrices[i]))
    trajectories.append(trans)

trajectories = np.array(trajectories)

# Read information about camera intrinsics
cameras = read_cameras_text("../../3D-data/cameras.txt")

widths, heights, focals, cxs,cys = [],[],[],[],[]
for i in range(len(cameras)):
    widths.append(cameras[i+1][2])
    heights.append(cameras[i+1][3])
    focals.append(cameras[i+1][4][0])
    cxs.append(cameras[i+1][4][1])
    cys.append(cameras[i+1][4][2])

widths = np.array(widths)
heights = np.array(heights)
focals = np.array(focals)
cxs = np.array(cxs)
cys = np.array(cys)

# Get (and save) depth maps from COLMAP, using their scripts/python/read_dense.py
depth_maps = []
for i in range(len(transformation_matrices)):
    depth_map = read_array('../../3D-data/depth_maps/frame-'+ f'{i+1:05}' + '.jpg.geometric.bin').astype(np.uint16)
    depth_maps.append(depth_map)
    if save_depth_maps == True:
        cv.imwrite('../../3D-data/depth_images/frame-'+ f'{i+1:05}' + '.png', depth_map)
        

depth_maps = np.array(depth_maps)
#%%
# Follow this tutorial for TSDF volume integration:
# http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html#TSDF-volume-integration
from time import asctime
print(asctime())

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length = 0.15 / 512.0, 
    sdf_trunc = 0.002, 
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(len(depth_maps)):
    if i % 10 == 0 or i == len(depth_maps) -1:
        print("Integrate image no {:d} into the volume.".format(i))
    color = o3d.io.read_image('../../3D-data/images/frame-' + f'{i+1:05}' + '.jpg')
    depth = o3d.io.read_image('../../3D-data/depth_images/frame-' + f'{i+1:05}' + '.png')
    #depth = o3d.cpu.pybind.geometry.Image(depth_map[i])
    #color = o3d.io.read_image("color/{:05d}.jpg".format(i))
    #depth = o3d.io.read_image("depth/{:05d}.png".format(i))
    #print(np.max(np.asarray(depth)))
    #print()
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    intrinsics = o3d.cpu.pybind.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(len(depth_maps[i][0]), len(depth_maps[i]),focals[i],focals[i],cxs[i],cys[i])
    #intrinsics.set_intrinsics(640, 480,focals[i],focals[i],cxs[i],cys[i])
    volume.integrate(
        rgbd,
        intrinsics,
        np.linalg.inv(trajectories[i])) # was transformation_matrices


print(asctime())
# Extract a mesh and visualize it
print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

print(mesh)
#%%
if visualize_mesh == True:
    visualize_cloud(mesh)
    '''
    o3d.visualization.draw_geometries([mesh],
                                  front=[0.5297, -0.1873, -0.8272],
                                  lookat=[2.0712, 2.0312, 1.7251],
                                  up=[-0.0558, -0.9809, 0.1864],
                                  zoom=0.47)

'''




















