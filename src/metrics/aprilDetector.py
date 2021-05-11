from dt_apriltags import Detector

import apriltag
import cv2 as cv2
from matplotlib import pyplot as plt
import math

def distance(pointA, pointB):
	(xA, yA) = pointA
	(xB, yB) = pointB
	distsqr = math.sqrt((xA-xB)*(xA-xB) + (yA-yB)*(yA-yB))
	return distsqr
def findPointsOtherLib(image):
	at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#print("shape of image: ", image.shape)
	results = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
	print("Results: ", results)
		# loop over the AprilTag detection result
	if(results == []):
		print("results in empty")
		return False
	r = results[0]
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))
	# draw the bounding box of the AprilTag detection
	cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	#print("line AB in 2d: ", distance(ptA, ptB))
	#print("line BC in 2d: ", distance(ptB, ptC))
	#print("line CD in 2d: ", distance(ptC, ptD))
	#print("line DA in 2d: ", distance(ptD, ptA))
	


	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	#cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	tagId = r.tag_id
	
	cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))
	print("[INFO] tag id: {}".format(tagId))
	# show the output image after AprilTag detection
	#s = cv2.resize(image, (700, 800))   
	#cv2.imshow("Image", s)
	#cv2.waitKey(0)
	print("before return")
	print("A: ", ptA)
	print("B: ", ptB)
	print("C: ", ptC)
	print("D: ", ptD)
	a=(0,0)
	b=(0,0)
	c=(0,0)
	d=(0,0)
	print("just before")
	#return (ptA, ptB, ptC, ptD, True)
	return True
	
	
	

def findPoints(image):
	#img = cv.imread('1small.png', cv.IMREAD_GRAYSCALE)
	#detector = apriltag.Detector()
	#result = detector.detect(img)
	print("[INFO] loading image...")
	#name = "2small.jpg"#"blackWhite.jpg"
	#name = "big1.jpg"
	#name = "1smallCropped.jpg"
	
	#name = "image.png"
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            
	#plt.imshow(th2,'gray')
	#plt.show()
	
	print("[INFO] detecting AprilTags...")
	options = apriltag.DetectorOptions(families="tag36h11")
	detector = apriltag.Detector(options)
	results = detector.detect(gray)
	print("results: ", results)
	print("[INFO] {} total AprilTags detected".format(len(results)))
	
	# loop over the AprilTag detection results
	r = results[0]
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))
	# draw the bounding box of the AprilTag detection
	cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	print("line AB in 2d: ", distance(ptA, ptB))
	print("line BC in 2d: ", distance(ptB, ptC))
	print("line CD in 2d: ", distance(ptC, ptD))
	print("line DA in 2d: ", distance(ptD, ptA))
	


	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	#cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	#cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))
	# show the output image after AprilTag detection
	s = cv2.resize(image, (700, 800))   
	#cv2.imshow("Image", s)
	#cv2.waitKey(0)
	
	return (ptB, ptC, ptD, ptA)
def main():
	#name = "../1smallNoPlant.jpg"#good
	#name = "../4smallNoPlant.png"#good: 
	name = "../1small.png"
	image = cv2.imread(name)
	findPointsOtherLib(image)
	
	
	
	

if __name__ == "__main__":
	main()
	

