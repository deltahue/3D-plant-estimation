
import open3d as o3d
import numpy as np

from utils import visualize_cloud, display_inlier_outlier, \
    save_point_cloud, save_mesh
    
from cluster.clustering_functions import read_config, show_clustering_result, \
    cluster_pc_HDBSCAN, extract_clusters

from mesh_generation import generate_mesh, smooth_mesh, remove_islands, remove_infs_nans

save_results = True
visualize    = True
plant = 'avocado'
#plant = 'luca2'
#plant = 'field'
#plant = 'palm'

if __name__== "__main__":
    
    assert (plant in ['luca2', 'avocado', 'field', 'palm'])
    
    if plant == 'avocado':
        pcd = o3d.io.read_point_cloud("/home/soley/3D-data/avocado_masked_cloud.ply")
        #pcd = o3d.io.read_point_cloud("/home/soley/3D-data/cropped_avo6.ply")
    elif plant == 'luca2':
        pcd = o3d.io.read_point_cloud("/home/soley/3D-data/luca2_masked_cloud.ply")
    elif plant == 'field':
        pcd = o3d.io.read_point_cloud("/home/soley/3D-data/cropped_field.ply")
    elif plant == 'palm':
        pcd = o3d.io.read_point_cloud("/home/soley/3D-data/cropped_palm.ply")
    
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
        rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=50, radius=0.1)
    elif plant == 'field':
        rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    elif plant == 'palm':
        rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=10, radius=0.1)
        
    
    print("Number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
    if visualize == True:
        display_inlier_outlier(voxel_down_cropped_pcd, ind)
        o3d.visualization.draw_geometries([rad_cl])
        
    
    if save_results == True and plant =='avocado':
        save_point_cloud('/home/soley/3D-data/filtered_avocado.ply', rad_cl)
        
    elif save_results == True and plant == 'luca2':
        save_point_cloud('/home/soley/3D-data/filtered_luca2.ply', rad_cl)
    elif save_results == True and plant == 'field':
        save_point_cloud('/home/soley/3D-data/filtered_field.ply', rad_cl)
    elif save_results == True and plant == 'palm':
        save_point_cloud('/home/soley/3D-data/filtered_palm.ply', rad_cl)
        
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
    for lab in range(len(clusters)-1):
        if visualize == True:
            o3d.visualization.draw_geometries([clusters[lab]])
        clusters[lab].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh = generate_mesh(clusters[lab], visualize = visualize)
        
        vertices = np.asarray(mesh.vertices)
        
        mesh = remove_islands(mesh, visualize == False)
        smooth = smooth_mesh(mesh, 10, visualize = visualize) 
        #smoother = remove_islands(smooth, visualize = visualize)
        
        final_mesh = remove_infs_nans(smooth)
        #hull, _ = final_mesh.compute_convex_hull()
        #o3d.visualization.draw_geometries([hull])
        
        if save_results == True:
            save_mesh('/home/soley/3D-data/mesh_label'+str(lab)+'.ply' , final_mesh)
    

    
    
    
    
    
    
