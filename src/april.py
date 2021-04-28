import colmap.scripts.python.read_write_model as col
from reconstruction3D.aprilDetector import findPointsOtherLib
import cv2 as cv2
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from dt_apriltags import Detector
import scale as sc  
import numpy as np


#a = (0,0,0)
#b = (0,0,0)
#c = (0,0,0)
#d = (0,0,0)

class April:
	def __init__(self, filePath):
		self.filePath =  filePath
		binPath = self.filePath + "/dense/sparse/images.bin" 
		imagesPath = self.filePath + "/dense/images/"
		point3Dpath = self.filePath + "/dense/sparse/points3D.bin" 
		images = col.read_images_binary(binPath)



		print("length of images: ", len(images))
		print(images[1].name)

		closest3DId_A = -1
		closest3DId_B = -1
		closest3DId_C = -1
		closest3DId_D = -1
	
		print("images length: ", len(images))
		at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
		for i in range(len(images)):
	
			distA = 1000000
			distB = 1000000
			distC = 1000000
			distD = 1000000
			print("----------------------------------------------------------------")
			print("name of image: ",images[i+1].name, " i: ", i+1)
		
			imageInteresting = images[i+1]
			image = cv2.imread(imagesPath + imageInteresting.name)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			results = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
			flag = True
			if(results == []):
				flag = False
			else:
				r = results[0]
				(A, B, C, D) = r.corners
				tagFamily = r.tag_family.decode("utf-8")
				tagId = r.tag_id
			
			if(flag):
				ind = 0
				for point in imageInteresting.xys:
					if(math.dist(point, A) < distA and imageInteresting.point3D_ids[ind] != -1):
						distA = math.dist(point, A)
						closest3DId_A = imageInteresting.point3D_ids[ind]
					if(math.dist(point, B) < distB and imageInteresting.point3D_ids[ind] != -1):
						distB = math.dist(point, B)
						closest3DId_B = imageInteresting.point3D_ids[ind]
					if(math.dist(point, C) < distC and imageInteresting.point3D_ids[ind] != -1):
						distC = math.dist(point, C)
						closest3DId_C = imageInteresting.point3D_ids[ind]
					if(math.dist(point, D) < distD and imageInteresting.point3D_ids[ind] != -1):
						distD = math.dist(point, D)
						closest3DId_D = imageInteresting.point3D_ids[ind]
					ind += 1
		print("id: ",closest3DId_A," ", closest3DId_B, " ", closest3DId_C, " ", closest3DId_D)	
		Aid3 = closest3DId_A
		Bid3 = closest3DId_B
		Cid3 = closest3DId_C
		Did3 = closest3DId_D
		self.points3d = col.read_points3d_binary(point3Dpath)

		self.a = self.points3d[Aid3].xyz
		self.b = self.points3d[Bid3].xyz
		self.c = self.points3d[Cid3].xyz
		self.d = self.points3d[Did3].xyz
		
		print("point3d for A: ", self.a)
		print("point3d for B: ", self.b)
		print("point3d for C: ", self.c) 
		print("point3d for D: ", self.d)
  	
	def findNormal(self):
		return np.cross(self.a - self.b, self.a - self.c)

	def findScale(self, squareSideLength):
		sideDist1 = math.dist(self.a, self.b)
		sideDist2 = math.dist(self.b, self.c)
		sideDist3 = math.dist(self.c, self.d)
		sideDist4 = math.dist(self.d, self.a)
		print("length calculated for side 1: ", sideDist1)
		print("length calculated for side 2: ", sideDist2)
		print("length calculated for side 3: ", sideDist3)
		print("length calculated for side 4: ", sideDist4)
		
		av = (sideDist1 + sideDist2 + sideDist3 + sideDist4) / 4
		scale =  squareSideLength / av
		return scale
		
	def showCorners(self):
		print("scale: ", scale)
		
		fig = plt.figure()
		fig.suptitle('3D reconstructed', fontsize=16)
		ax = fig.gca(projection='3d')

		s=0
		print("length of points3d",len(points3d))
		xs = []
		ys = []
		zs = []
		for key in points3d:
			s+=1
			xs.append(self.points3d[key].xyz[0])
			ys.append(self.points3d[key].xyz[1])
			zs.append(self.points3d[key].xyz[2])

		ax.plot(xs, ys, zs, 'b.')
		ax.plot(self.a[0], self.a[1], self.a[2], 'r.')
		ax.plot(self.b[0], self.b[1], self.b[2], 'r.')
		ax.plot(self.c[0], self.c[1], self.c[2], 'r.')
		ax.plot(self.d[0], self.d[1], self.d[2], 'r.')
		
		plt.show()
	
	
	

		

	
