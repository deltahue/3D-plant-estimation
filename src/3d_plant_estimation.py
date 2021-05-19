
import open3d as o3d
import numpy as np

from utils import visualize_cloud, display_inlier_outlier, create_bounding_box, \
    save_point_cloud, save_mesh
    
from cluster.clustering_functions import read_config, show_clustering_result, \
    cluster_pc_HDBSCAN, extract_clusters

from mesh_generation import generate_mesh, smooth_mesh, remove_islands, remove_infs_nans

save_results = True
visualize    = True
#plant = 'avocado'
plant = 'avocado'

if __name__== "__main__":
    if plant == 'avocado':
        pcd = o3d.io.read_point_cloud("../../3D-data/avocado_masked_cloud.ply")
    elif plant == 'luca2':
        pcd = o3d.io.read_point_cloud("../../3D-data/luca2_masked_cloud.ply")
    
    if visualize == True:
        visualize_cloud(pcd)
        
    cropped_pcd = pcd
    
    #Downsample and remove outliers from cropped point cloud
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
    if visualize == True:
        o3d.visualization.draw_geometries([voxel_down_cropped_pcd])
    
    
    print("Radius oulier removal")
    if plant == 'avocado':
        rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=20, radius=0.1)
    elif plant == 'luca2':
        rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=60, radius=0.1)
        
    
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
    
    if visualize == True:
        show_clustering_result(pc, labels)

    clusters = extract_clusters(pc, labels, config)
    # Array of clusters indexed by labels
    
    # Generate and visualize mesh for each cluster
    # TODO: Identify leaves and only generate mesh for them
    for lab in range(len(clusters)-1):
        if visualize == True:
            o3d.visualization.draw_geometries([clusters[lab]])
        clusters[lab].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh = generate_mesh(clusters[lab], visualize = False)
        
        vertices = np.asarray(mesh.vertices)
        print(np.max(np.isnan(vertices)))
        
        smooth = smooth_mesh(mesh, 10, visualize = False) 
        smoother = remove_islands(smooth, visualize = visualize)
        
        final_mesh = remove_infs_nans(smoother)
        
        if save_results == True:
            save_mesh('../../3D-data/mesh_label'+str(lab)+'.ply' , final_mesh)
    

    
    
    
    
    
    
