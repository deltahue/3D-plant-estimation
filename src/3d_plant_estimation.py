
import open3d as o3d
import numpy as np
import os



import metrics.april as ap
import metrics.surfaceEstimation as su
import metrics.normalEstimation as fn 
import metrics.angleEstimation as an
import labeling.classifier as cl

import matplotlib.pyplot as plt


from utils import visualize_cloud, display_inlier_outlier, create_bounding_box, \
    save_point_cloud, save_mesh
    
from cluster.clustering_functions import read_config, show_clustering_result, \
    cluster_pc_HDBSCAN, extract_clusters

from mesh_generation import generate_mesh, smooth_mesh

save_results = True
visualize    = False

if __name__== "__main__":

    pathRoot = "../../../"#'../../3D-data/
    holeMeshName = "fused.ply"
    croppedPcdRawName = 'cropped_pcd_raw_avocado.ply'
    croppedPcdFilteredName = 'cropped_pcd_filtered_avocado_rad20_0p1.ply'
    hdbscanPath = "./cluster/config/hdbscan_config.yaml"
    colmapFolderName = "luca2"
    apriltagSide = 16

    
    pcd = o3d.io.read_point_cloud(pathRoot + holeMeshName)
    
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
        save_point_cloud(pathRoot + croppedPcdRawName, cropped_pcd)
    
    
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
        save_point_cloud(pathRoot + croppedPcdFilteredName, rad_cl)
        
    # read files
    # TODO IO function with try statement
    config = read_config(hdbscanPath)
    path = config['path']
    pc = o3d.io.read_point_cloud(path)
    print(pc)
    if config['show_raw']:
        o3d.visualization.draw_geometries([pc])

    print("here-1")
    labels = cluster_pc_HDBSCAN(pc, config)

    print("here0")
    
    if visualize == True:
        show_clustering_result(pc, labels)
    print("here1")


    clusters = extract_clusters(pc, labels, config)
    # Array of clusters indexed by labels

    print("here2")
    
    # Generate and visualize mesh for each cluster
    # TODO: Identify leaves and only generate mesh for them
    for lab in range(len(clusters)-1):
        if visualize == True:
            o3d.visualization.draw_geometries([clusters[lab]])
        print("here3")
        clusters[lab].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("here4")
        mesh = generate_mesh(clusters[lab], visualize = visualize)
        print("here5")
        smooth = smooth_mesh(mesh, 10, visualize) 
        print("here6")
        if save_results == True:
            save_mesh(pathRoot + '/organs/mesh_label'+str(lab)+'.ply' , smooth)

    metrics = False
    if metrics:

        ##############################################
        ################## SCALE #####################
        ##############################################
        colmapOutputPath = pathRoot + colmapFolderName
        april = ap.April(colmapOutputPath)
        scale = april.findScale(apriltagSide)

        ##############################################
        ################## HEIGHT ####################
        ##############################################
        meshPath = pathRoot + holeMeshName
        normal = april.findNormal()
        croppedMesh = pathRoot + croppedPcdFilteredName
        heightnotScaled = fn.getHeight(croppedMesh, normal, april.a)
        print("heigh not scaled: ", heightnotScaled)
        heightScaled = heightnotScaled * scale
        print("height is: ", heightScaled)

        ##############################################
        ##########        labeling        ############
        ##############################################
        classifier = cl.Classifier("./data/training")

        ##############################################
        ########## Surface area eand angle############
        ##############################################
        leafCnt = 0
        stemCnt = 0
        for filename in os.listdir(pathRoot + "/organs/"):
            #need a for here to go through all of the organs
            organPath = pathRoot + "organs/" + filename
            ##############################################
            ##################   type    #################
            ##############################################
            type = -1
            type = classifier.classify(organPath)
            if(type == 1):
                os.rename(organPath, pathRoot + "organs/" + "leaf" + leafCnt + ".ply")
                leafCnt += 1
            if(type == 0):
                os.rename(organPath, pathRoot + "organs/" + "stem" + stemCnt + ".ply")
                stemCnt += 1
            
            sums = su.findAreaOfTop(organPath)
            #should only do this for leaves
            arr = an.findAngle(organPath, normal)
            leafSurface = sums * scale
            print("leaf area is: ", leafSurface)
        
        plt.hist(arr, bins=30)
        plt.show()

 


	    





    

    
    
    
    
    
    
