# Author David Helm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
#######################################
# UTILS
#######################################
# import cloud
def import_3d_cloud(path):
    #return o3d.io.read_triangle_mesh(path)
    return o3d.io.read_point_cloud(path)


# cleaning
# TODO: Gridsearch for parameters
def remove_outlier(point_cloud, debug=False):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=40,
                                                     std_ratio=0.1)
    if debug:
        display_inlier_outlier(point_cloud, ind)
    return cl


def voxel_downsampling(point_cloud, vox_size=0.05):
    return point_cloud.voxel_down_sample(voxel_size=vox_size)


# clustering methods
def dbscan(point_cloud, debug=False):
    # TODO Gridsearch for param

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            point_cloud.cluster_dbscan(eps=0.2, min_points=30, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([point_cloud])

    return point_cloud


def spectral_clustering(point_cloud):

    return 'Todo'


def hierarchical_clustering(point_clustering):

    return 'Todo'




# visualize
def show_point_cloud(point_cloud, title='3D-Vision-Project'):
    o3d.visualization.draw_geometries([point_cloud], window_name=title)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (green): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#######################################
# PIPELINES
#######################################

def preprocessing(point_cloud, debug=True):
    point_cloud = voxel_downsampling(point_cloud, 0.05)
    if debug:
        show_point_cloud(point_cloud, 'Downsampled')
    #point_cloud = remove_outlier(point_cloud, debug)
    if debug:
        show_point_cloud(point_cloud, 'Outliers Removed')

    return point_cloud


def clustering(point_cloud, debug=False):
    point_cloud = dbscan(point_cloud, debug)

if __name__ == '__main__':
    #path = '../../data/processed/init_testing/fused.ply'
    path = '../../data/processed/meshed/meshed-poisson.ply'
    #path = '../../data/processed/init_testing/merged_avocado_V3_Opensfm.ply'


    pc = import_3d_cloud(path)
    show_point_cloud(pc, 'Raw PC')
    #processed = preprocessing(pc, debug=True)
    #show_point_cloud(pc, 'Processed PC')
    clustering(pc, True)



