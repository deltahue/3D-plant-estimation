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
    print("end get XY")
    return np.hstack([arr, carr])

def create_voxel_grid(pointcloud, div):
    voxel_size = max(pointcloud.get_max_bound() - pointcloud.get_min_bound()) / div
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, voxel_size=voxel_size)


# visualization
def show_clustering_result(pointcloud, labels):
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
    print("config: ", config)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=config['min_cluster_size'],
                                gen_min_span_tree=config['gen_min_span_tree'],
                                cluster_selection_method=config['cluster_selection_method'],
                                min_samples=config['min_samples'])
    print("do hbscan")
    clusterer.fit(arr)
    return clusterer

# main wrapper
def cluster_pc_HDBSCAN(pointcloud, config):
    print("start cluster")
    np_6D = get_XYZHSV_np(pointcloud)
    print("mid clusterer")
    hdbscan = do_HDBSCAN(np_6D, config['hdbscan'])

    labels = hdbscan.labels_
    print("Found " + str(len(np.unique(labels))-1) + " clusters!")

    if config['hdbscan']['hdbscan_debug']:
        hdbscan.condensed_tree_.plot(select_clusters=True)
        plt.show()
    return labels


def extract_clusters(pointcloud, labels, config):
    labels_classes = np.unique(labels)
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    print(points.shape)
    clusters = {}
    for label in labels_classes:
        c_points = points[labels == label, :]
        c_colors = colors[labels == label, :]

        cluster_pc = o3d.geometry.PointCloud()
        cluster_pc.points = o3d.utility.Vector3dVector(c_points)
        cluster_pc.colors = o3d.utility.Vector3dVector(c_colors)
        clusters[label] = cluster_pc

    if config['show_extracted_clusters']:
        for label in labels_classes:
            o3d.visualization.draw_geometries([clusters[label]])

    return clusters