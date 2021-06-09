
import open3d as o3d
import numpy as np
import os
import sys

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


from mesh_generation import generate_mesh, smooth_mesh, remove_islands, remove_infs_nans


save_results = True
visualize    = False

if __name__== "__main__":

    args = sys.argv[1:]
    if(args[0] == "luca2"):
        from config_luca2 import *
        plantName = "luca2"
    if(args[0] == "avo_6"):
        from config_avo_6 import *
        plantName = "avocado"
        
    # Crop the point cloud and export results
    cropped_pcd = o3d.io.read_point_cloud(pathRoot + croppedMeshName)
    
    if visualize == True:
        visualize_cloud(cropped_pcd)
    
    #Downsample and remove outliers from cropped point cloud
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
    if visualize == True:
        o3d.visualization.draw_geometries([voxel_down_cropped_pcd])
    
   
    print("Radius oulier removal")
    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    #if plant == 'avocado':
    #    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=20, radius=0.1)
    #elif plant == 'luca2':
    #    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=50, radius=0.1)
    #elif plant == 'field':
    #    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    #elif plant == 'palm':
    #    rad_cl, ind = voxel_down_cropped_pcd.remove_radius_outlier(nb_points=10, radius=0.1)
      
    print("Number of outliers is: " + str(len(np.asarray(voxel_down_cropped_pcd.points)) - len(np.asarray(rad_cl.points))) + '/' + str(len(np.asarray(voxel_down_cropped_pcd.points))))
    if visualize == True:
        display_inlier_outlier(voxel_down_cropped_pcd, ind)
        o3d.visualization.draw_geometries([rad_cl])

     
    if save_results == True:
        save_point_cloud(pathRoot + croppedPcdFilteredName, rad_cl)
        
    # read files
    # TODO IO function with try statement
    config = read_config(hdbscanPath)
    path = pathRoot + croppedPcdFilteredName #config['path']
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

    for filename in os.listdir(pathOrgans+ plantName):
        os.remove(pathOrgans+ plantName + "/" + filename)
    
    # Generate and visualize mesh for each cluster
    for lab in range(len(clusters)-1):
        if visualize == True:
            o3d.visualization.draw_geometries([clusters[lab]])
        clusters[lab].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh = generate_mesh(clusters[lab], visualize = visualize)
        
        vertices = np.asarray(mesh.vertices)
        
        mesh = remove_islands(mesh, visualize = visualize)
        smooth = smooth_mesh(mesh, 10, visualize = visualize) 
        #smoother = remove_islands(smooth, visualize = visualize)
        
        #removing the old meshes
        final_mesh = remove_infs_nans(smooth)
        if not os.path.exists(pathOrgans):
            os.makedirs(pathOrgans)
        
        if save_results == True:
            save_mesh(pathOrgans+ plantName + "/mesh_label"+str(lab)+'.ply' , smooth)

    metrics = True
    if metrics:

        ##############################################
        ################## SCALE #####################
        ##############################################
        colmapOutputPath = pathRoot2 + colmapFolderName
        april = ap.April(colmapOutputPath)
        scale = april.findScale(apriltagSide)

        ##############################################
        ################## HEIGHT ####################
        ##############################################
        normal = april.findNormal()
        croppedMesh = pathRoot + croppedPcdFilteredName
        heightnotScaled = fn.getHeight(croppedMesh, normal, april.a)
        print("heigh not scaled: ", heightnotScaled)
        heightScaled = heightnotScaled * scale
        print("scale: ", scale)
        print("height is: ", heightScaled)

        ##############################################
        ##########        labeling        ############
        ##############################################
        classifier = cl.Classifier(pathOrganClass)

        ##############################################
        ########## Surface area eand angle############
        ##############################################
        leafCnt = 0
        stemCnt = 0
        for filename in os.listdir(pathOrgans+ plantName):
            #need a for here to go through all of the organs
            organPath = pathOrgans+ plantName + "/" + filename
            #print("organPath: ", organPath)
            ##############################################
            ##################   type    #################
            ##############################################
            type = -1
            type = classifier.classify(organPath)
            #print("leaf number: ", leafCnt)
            if(type == 1):
                os.rename(organPath, pathOrgans+ plantName + "/" + "leaf" + str(leafCnt) + ".ply")
                leafCnt += 1
            if(type == 0):
                os.rename(organPath, pathOrgans+ plantName + "/" + "stem" + str(stemCnt) + ".ply")
                stemCnt += 1

        for filename in os.listdir(pathOrgans+ plantName):
            organPath = pathOrgans+ plantName + "/" + filename    
            sums = su.findAreaOfTop(organPath)
            #should only do this for leaves
            arr = an.findAngle(organPath, normal)
            leafSurface = sums * scale
            print("leaf area is: ", leafSurface)
        
        plt.hist(arr, bins=30)
        plt.show()

 


	    





    

    
    
    
    
    
    
