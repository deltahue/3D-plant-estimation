
import open3d as o3d
import numpy as np

from utils import visualize_cloud, display_inlier_outlier, create_bounding_box, \
    save_point_cloud, save_mesh
    
from cluster.clustering_functions import read_config, show_clustering_result, \
    cluster_pc_HDBSCAN, extract_clusters

from mesh_generation import generate_mesh, smooth_mesh

save_results = True
visualize    = False

if __name__== "__main__":
    
    pcd = o3d.io.read_point_cloud("../../3D-data/avocado_pcd.ply")
    
    if visualize == True:
        visualize_cloud(pcd)
        
    #Crop the point cloud to just the plant
    cen = np.array([1.5,1,2])
    ext = np.array([5,5,4])
    rotations = np.array([0,30,0])
    
    bbox, aabbox = create_bounding_box(cen, ext, rotations)
    
    cropped_pcd = pcd.crop(bbox)
    
    if visualize == True:
        o3d.visualization.draw_geometries([cropped_pcd, bbox ,aabbox])
        o3d.visualization.draw_geometries([cropped_pcd])
        
    # Crop the point cloud and export results
    cropped_pcd = pcd.crop(bbox)
    
    if visualize == True:
        visualize_cloud(cropped_pcd)
    
    if save_results == True:
        save_point_cloud('../../3D-data/cropped_pcd_raw_avocado.ply', cropped_pcd)
    
    
    #Downsample and remove outliers from cropped point cloud
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
    if visualize == True:
        o3d.visualization.draw_geometries([voxel_down_cropped_pcd])
    
    
    print("Radius oulier removal")
    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=20, radius=0.1) #80 for leaf, 60 for newplant, 20 avo
    print("number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
    if visualize == True:
        display_inlier_outlier(voxel_down_cropped_pcd, ind)
        o3d.visualization.draw_geometries([rad_cl])
        
    
    if save_results == True:
        save_point_cloud('../../3D-data/cropped_pcd_filtered_avocado_rad20_0p1.ply', rad_cl)
        
    # read files
    # TODO IO function with try statement
    config = read_config("./cluster/config/hdbscan_config.yaml")
    path = config['path']
    pc = o3d.io.read_point_cloud(path)
    print(pc)
    if config['show_raw']:
        o3d.visualization.draw_geometries([pc])


    labels = cluster_pc_HDBSCAN(pc, config)
    show_clustering_result(pc, labels)

    clusters = extract_clusters(pc, labels, config)
    # Array of clusters indexed by labels
    
    # Generate and visualize mesh for each cluster
    # TODO: Identify leaves and only generate mesh for them
    for lab in range(len(clusters)-1):
        o3d.visualization.draw_geometries([clusters[lab]])
        clusters[lab].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh = generate_mesh(clusters[lab], True)
        # TODO: Find bug in smoothing
        #smooth = smooth_mesh(clusters[lab], True) 
        
        if save_results == True:
            #TODO: find bug, probably in savestring
            save_mesh('../../3D-data/mesh_label'+str(lab)+'.ply' , mesh)
    

    
    
    
    
    
    
