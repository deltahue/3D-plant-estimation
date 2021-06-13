
import open3d as o3d
import numpy as np
import os
import sys
import copy
import math

import metrics.april as ap
import metrics.surfaceEstimation as su
import metrics.normalEstimation as fn 
import metrics.angleEstimation as an
import labeling.classifier as cl

import matplotlib.pyplot as plt

import argparse


from utils import visualize_cloud, display_inlier_outlier, create_bounding_box, \
    save_point_cloud, save_mesh
    
from cluster.clustering_functions import read_config, show_clustering_result, \
    cluster_pc_HDBSCAN, extract_clusters


from mesh_generation import generate_mesh, smooth_mesh, remove_islands, remove_infs_nans


save_results = True
visualize    = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__== "__main__":

    args = sys.argv[1:]
    

    parser = argparse.ArgumentParser(description='name of plant')
    parser.add_argument('-plantName', metavar='name', type=str, help='the name of the plant')
    parser.add_argument('-show', metavar='show', type=str2bool, default=False, help='show the results in the terminal')
    args = parser.parse_args()
    #print(args.show)
    #print(args.plantName)

    if(args.plantName == "luca2"):
        from config_luca2 import *
        plantName = "luca2"
    elif(args.plantName == "avo_6"):
        from config_avo_6 import *
        plantName = "avocado"
    elif(args.plantName == "palm"):
        from config_palm import *
        plantName = "palm"
    elif(args.plantName == "field"):
        from config_field import *
        plantName = "field"
    
        
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

    if config['show_raw']:
        o3d.visualization.draw_geometries([pc])

    print("finding the labels")
    labels = cluster_pc_HDBSCAN(pc, config)

    if visualize == True:
        show_clustering_result(pc, labels)


    clusters = extract_clusters(pc, labels, config)
    # Array of clusters indexed by labels

    if not os.path.exists(pathOrgans):
        os.makedirs(pathOrgans)
    if not os.path.exists(pathOrgans+ plantName):
        os.makedirs(pathOrgans+ plantName)

    for filename in os.listdir(pathOrgans+ plantName):
        os.remove(pathOrgans+ plantName + "/" + filename)
    
    # Generate and visualize mesh for each cluster
    #
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
        if not os.path.exists(pathRoot + "results"):
            os.makedirs(pathRoot + "results")
        file1 = open(pathRoot + "results" + "/res.txt","a")
        ##############################################
        ################## SCALE #####################
        ##############################################
        colmapOutputPath = pathRoot2 + colmapFolderName
        pcdFull = o3d.io.read_point_cloud(pathRoot + MeshOrigName)
        april = ap.April(colmapOutputPath, plantName, pcdFull, args.show, file1)
        scale = april.findScale(apriltagSide)
        debug = False
        if(debug):
            #pcd = o3d.io.read_point_cloud(pathRoot + MeshOrigName)
            pcd = o3d.io.read_point_cloud(pathRoot + croppedPcdFilteredName)
            pcl = o3d.geometry.PointCloud()
            points = np.array([april.a, april.b, april.c, april.d])
            #print("shape of points: ", points.shape)
            pcl.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1, 0.706, 0])
            pcl.paint_uniform_color([1, 0, 0])

            all = o3d.geometry.PointCloud()
            allPoints = np.concatenate((points, pcd.points), axis=0)
            all.points = o3d.utility.Vector3dVector(allPoints)
            o3d.io.write_point_cloud("allPoints.ply", all)

            box = pcd.get_axis_aligned_bounding_box()
            mesh_r = copy.deepcopy(pcd)
            #print(x.shape)
            # new axes:
            nnx, nny, nnz = april.x, april.y, april.z #new_xaxis, new_yaxis, new_zaxis
            R = np.hstack((april.x, april.y, april.z))
            R = R.reshape(3,-1)
            #R = mesh.get_rotation_matrix_from_xyz(np.array(x, y, z))
            mesh_r.rotate(R, center=(0, 0, 0))
            box = mesh_r.get_axis_aligned_bounding_box()
            minPoint = box.get_min_bound()
            maxPoint = box.get_max_bound()
            distz = abs(maxPoint[2]-minPoint[2])

            print("min: ", box.get_min_bound())
            print("max: ", box.get_max_bound())
            print("distz: ", distz)
            o3d.visualization.draw_geometries([box, mesh_r])


        

        ##############################################
        ################## HEIGHT ####################
        ##############################################
        normalMethod = "APRIL" #  "APRIL"
        normal = april.findNormal(normalMethod)
        croppedMesh = pathRoot + croppedPcdFilteredName
        heightnotScaled = fn.getHeight(croppedMesh, normal, april.a)
        heightScaled = heightnotScaled * scale

        a = "height not scaled: "+ str(heightnotScaled)
        b = "scale: "+ str(scale)
        c = "estimated height is: "+ str(heightScaled)
        if(args.show):
            print(a)
            print(b)
            print(c)
        else:
            file1.write(a + "\n")
            file1.write(b + "\n")
            file1.write(c + "\n")

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
        
        
        length = len(os.listdir(pathOrgans+ plantName))

        for filename in os.listdir(pathOrgans+ plantName):
            organPath = pathOrgans+ plantName + "/" + filename    
            sums = su.findAreaOfTop(organPath)
            #should only do this for leaves
            arr = an.findAngle(organPath, normal)
            #print("sums: ", sums)
            leafSurface = sums * scale *scale
            a = "leaf area of "+ filename + " is: "+ str(leafSurface)
            if(args.show):
                print(a)
            else:
                file1.write(a + "\n")
            plt.clf()
            plt.hist(arr, bins=30)
            plt.xlabel('angle')
            plt.title('mesh name: ' + filename)

            if(args.show):
                plt.show()
            else:
                plt.savefig(pathRoot + "results/" +"histo_" +filename +'.png')


 


	    





    

    
    
    
    
    
    
