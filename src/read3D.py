import colmap.scripts.python.read_write_model as col
from reconstruction3D.aprilDetector import findPointsOtherLib
import cv2 as cv2
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from dt_apriltags import Detector
import april as ap
import surfaceEstimation as su
import normalEstimation as fn 
import angleEstimation as an
import numpy as np 
import matplotlib.pyplot as plt


	
def main():

	path = "./luca2"
	april = ap.April(path)
	scale = april.findScale(16)
	print("**********************************************")
	print("the scale is: ", scale)
	print("**********************************************")
	meshPath = path + "/meshed-poisson.ply"
	#normUp = sc.findNormal()
	#normal = (0,0,1)
	#this leaf is not the leaf of this mesh
	leafPath = "leafCropped.ply"
	sums = su.findAreaOfTop(leafPath)
	print("***********************************************")
	print("sum of the area of the triangles: ", sums * scale)
	print("***********************************************")
	normal = april.findNormal()
	#np.random.seed(42)
	#x = np.random.normal(size=1000)
	arr = an.findAngle(leafPath, normal)
	plt.hist(arr, bins=30)
	plt.show()
	print("normal bef: ", normal)
	#normal *= -1
	print("normal af: ", normal)
	#norVec = get_arrow(origin=[0,0,0], vec=normal*10)
	#mesh = o3d.io.read_triangle_mesh(meshPath)
	#o3d.visualization.draw_geometries([norVec,mesh])
	print("***********************************************")
	print("normal: ", normal)
	print("***********************************************")
	nor2 = -1 * normal
	#fn.drawNormal(meshPath, normal)
	#croppedMesh = path + "/cropped_pcd_filtered_newplant.ply"
	croppedMesh =  path + "/croppedByHand.ply"
	heightnotScaled = fn.getHeight(croppedMesh, normal, april.a)
	print("heigh not scaled: ", heightnotScaled)
	heightScaled = heightnotScaled * scale
	print("height is: ", heightScaled)
	#add also the leaf surface but the segmented version is not yet available
	print("leaf area is: ", sums * scale)
	#doing the angle for one single leaf but thee segmented version is not yet available
	#
	
	#plt.hist(x, density=True, bins=30)  # density=False would make counts
	#plt.show()


	
	
	
	
			
if __name__ == "__main__":
	main()

