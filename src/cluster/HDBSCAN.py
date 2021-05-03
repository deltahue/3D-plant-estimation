"""
3D Vision 2021
"""
import open3d as o3d
import yaml
import numpy as np
import hdbscan
import matplotlib.pyplot as plt

import matplotlib.colors as C



# helpers
def read_config(config_file):
    # Read config from file
    try:
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("The config file ({config_file}) you specified does not exist.")
    return config


def get_XYZHSV_np(pointcloud):
    arr = np.asarray(pointcloud.points)
    carr = np.asarray(pointcloud.colors)
    carr = C.rgb_to_hsv(carr)
    return np.hstack([arr, carr])


def downsample(pointcloud, div):
    voxel_size = max(pc.get_max_bound() - pc.get_min_bound()) / div
    return pointcloud.voxel_down_sample(voxel_size=voxel_size)


def create_voxel_grid(pointcloud, div):
    voxel_size = max(pointcloud.get_max_bound() - pointcloud.get_min_bound()) / div
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, voxel_size=voxel_size)


# visualization
def show_clustering_result(pointcloud, labels):
    # TODO assert length
    n = len(np.unique(labels))-1
    colors = plt.get_cmap("tab20")(labels / (n if n > 0 else 1))
    # make noise grey
    colors[labels<0] = [0.7,0.7,0.7,0.7]

    # copy pointcloud for visualization
    pc_vis = o3d.geometry.PointCloud(pointcloud)
    pc_vis.colors = o3d.utility.Vector3dVector(colors[:, :3])

    voxel_grid = create_voxel_grid(pc_vis, 200)
    o3d.visualization.draw_geometries([voxel_grid])


# clustering
def do_HDBSCAN(arr, config):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=config['min_cluster_size'],
                                gen_min_span_tree=config['gen_min_span_tree'],
                                cluster_selection_method=config['cluster_selection_method'],
                                min_samples=config['min_samples'])
    clusterer.fit(arr)
    return clusterer

# main wrapper
def cluster_pc_HDBSCAN(pointcloud, config):
    np_6D = get_XYZHSV_np(pointcloud)
    hdbscan = do_HDBSCAN(np_6D, config['hdbscan'])

    labels = hdbscan.labels_
    print("Found " + str(len(np.unique(labels))-1) + " clusters!")

    # TODO debug plotting
    return labels


if __name__== "__main__":
    # read files
    # TODO IO function with try statement
    config = read_config("./config/hdbscan_config.yaml")
    path = config['path']
    pc = o3d.io.read_point_cloud(path)
    print(pc)
    if config['show_raw']:
        o3d.visualization.draw_geometries([pc])

    # downsample
    pc_ds = downsample(pc, config['downsample_div'])
    print("Downsampled to " +str(pc_ds))
    # hdbscan
    labels = cluster_pc_HDBSCAN(pc_ds, config)

    show_clustering_result(pc_ds, labels)



