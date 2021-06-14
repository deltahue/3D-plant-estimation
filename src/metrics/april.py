import colmap.scripts.python.read_write_model as col
import cv2 as cv2
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from dt_apriltags import Detector 
import numpy as np
import metrics.ransac as ran
import os


class April:
	def __init__(self, filePath, _plantName, _pcd, _show, _file1):
		self.filePath =  filePath
		if os.path.exists(self.filePath + "/dense/sparse/0/images.bin" ):
			binPath = self.filePath + "/dense/sparse/0/images.bin" 
			imagesPath = self.filePath + "/dense/images/"
			point3Dpath = self.filePath + "/dense/sparse/0/points3D.bin" 
		else:
			binPath = self.filePath + "/dense/sparse/images.bin" 
			imagesPath = self.filePath + "/dense/images/"
			point3Dpath = self.filePath + "/dense/sparse/points3D.bin" 
		images = col.read_images_binary(binPath)
		self.plantName = _plantName
		self.pcd = _pcd
		self.show = _show
		self.file1 = _file1
		self.x = None
		self.y = None
		self.z = None

		closest3DId_A = -1
		closest3DId_B = -1
		closest3DId_C = -1
		closest3DId_D = -1
		closest3DId_E = -1
		distA = 10000000
		distB = 10000000
		distC = 10000000
		distD = 10000000
		distE = 10000000
		at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
		if(self.plantName == "field"):
				at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=20.0, quad_sigma=1, refine_edges=1, decode_sharpening=1, debug=0)
			
		for i in range(len(images)):
	
			print("----------------------------------------------------------------")
			print("detecting apriltag corners of image: ",images[i+1].name, " i: ", i+1)
		
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
				E = r.center
				tagFamily = r.tag_family.decode("utf-8")
				tagId = r.tag_id
			if(flag):
				print("corners of image ", images[i+1].name, " were detected")
			else:
				print("NO corners for image ", images[i+1].name, " were detected")
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
					if(math.dist(point, E) < distE and imageInteresting.point3D_ids[ind] != -1):
						distE = math.dist(point, E)
						closest3DId_E = imageInteresting.point3D_ids[ind]
					ind += 1
		a = "closest 2d point having matching 3D point is "+ str(distA) + " away from april from first apriltag corner"
		b = "closest 2d point having matching 3D point is "+ str(distB) + " away from april from second apriltag corner"
		c = "closest 2d point having matching 3D point is "+ str(distC) + " away from april from third apriltag corner"
		d = "closest 2d point having matching 3D point is "+ str(distD) + " away from april from fourth apriltag corner"
		if(self.show):
			print(a)
			print(b)
			print(c)
			print(d)
		else:
			self.file1.write(a+ "\n")
			self.file1.write(b+ "\n")
			self.file1.write(c+ "\n")
			self.file1.write(d+ "\n")
		Aid3 = closest3DId_A
		Bid3 = closest3DId_B
		Cid3 = closest3DId_C
		Did3 = closest3DId_D
		Eid3 = closest3DId_E
		self.points3d = col.read_points3D_binary(point3Dpath)
		

		self.a = self.points3d[Aid3].xyz
		self.b = self.points3d[Bid3].xyz
		self.c = self.points3d[Cid3].xyz
		self.d = self.points3d[Did3].xyz
		self.e = self.points3d[Eid3].xyz


	def findNormal(self, normalMethod):
		if(normalMethod == "APRIL"):
			print(self.a)
			return np.cross(self.a - self.b, self.a - self.c)
		elif(normalMethod == "RANSAC"):
			aa, bb, dd = ran.findPlane(self.pcd)
			print("aa: ", aa,"bb: ", bb, "dd: ", dd)
			x1 = 50
			y1 = 50
			z1 = aa * x1 + bb * y1 + dd

			x2 = 100
			y2 = 100
			z2 = aa * x2 + bb * y2 + dd

			x3 = 2
			y3 = 2
			z3 = aa * x3 + bb * y3 + dd

			self.point1 = np.array([x1, y1, z1])
			print(self.point1)
			print("shape: ", self.point1.shape)
			self.point2 = np.array([x2, y2, z2])
			self.point3 = np.array([x3, y3, z3])
			print("point1: ", self.point1, "point2: ", self.point2, "point3: ", self.point3)
			print("normal: ", np.cross(self.point1 - self.point2, self.point1 - self.point3))
			return np.cross(self.point1 - self.point2, self.point1 - self.point3)

			


	def findScale(self, squareSideLength):
		sideDist1 = math.dist(self.a, self.b)
		sideDist2 = math.dist(self.b, self.c)
		sideDist3 = math.dist(self.c, self.d)
		sideDist4 = math.dist(self.d, self.a)

		a1 = "--------------------------------------------------------------------"
		a = "non scaled length calculated for first side of apriltag: "+ str(sideDist1)
		b = "non scaled length calculated for second side of apriltag: "+ str(sideDist2)
		c = "non scaled length calculated for third side of apriltag: "+ str(sideDist3)
		d = "non scaled length calculated for fourth side of apriltag: "+ str(sideDist4)
		d1 = "--------------------------------------------------------------------"
		if(self.show):
			print(a1)
			print(a)
			print(b)
			print(c)
			print(d)
			print(d1)
		else:
			self.file1.write(a1+ "\n")
			self.file1.write(a+ "\n")
			self.file1.write(b+ "\n")
			self.file1.write(c+ "\n")
			self.file1.write(d+ "\n")
			self.file1.write(d1+ "\n")

		diag1 = math.dist(self.a, self.e)
		diag2 = math.dist(self.b, self.e)
		diag3 = math.dist(self.c, self.e)
		diag4 = math.dist(self.d, self.e)
		
		av = (sideDist1 + sideDist2 + sideDist3 + sideDist4) / 4
		scale =  squareSideLength / av

		self.x = np.array(self.a - self.b)
		self.y = np.array(self.a - self.d)	
		self.z = np.cross(self.a - self.b, self.a - self.c)
		return scale


	def showCorners(self):
		fig = plt.figure()
		fig.suptitle('3D reconstructed', fontsize=16)
		ax = fig.gca(projection='3d')

		s=0
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
