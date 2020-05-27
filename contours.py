import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

def get_max_contour(image, bg, threshold= 25, gaussian_kernel_size= (11, 11), 
					 erode_iter= 3, dilate_iter= 3):
		'''
		Function to get the contour of a hand. Assumes that hand
		has the maximum contour area.

		Args:
		image: Image. Image to process
		bg: background image
		threshold: threshold for gaussian blur
		gausssian_kernel_size: int. Kernel size to apply gaussian blur
		erode_iter: int. Number of iterations to erode
		dilate_iter: int. Number of iterations to dilate

		Returns:
		c: maximum contour in the given region. None if no contour present
		'''
		diff = cv2.absdiff(bg.astype('uint8'), image)
		#Change to grayscale
		gray = cv2.cvtColor(diff , cv2.COLOR_BGR2GRAY)

		#Remove high frequency noise
		gray = cv2.GaussianBlur(gray, gaussian_kernel_size, 0)

		#Identifying foreground and background
		thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.erode(thresh, None, iterations= erode_iter)
		thresh = cv2.dilate(thresh, None, iterations= dilate_iter)
		#Detecting conoturs
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		#If contour found, find max of contours. That will be the hand
		if len(cnts) == 0:
			return None
		else:
			c = max(cnts, key=cv2.contourArea)
			return thresh, c

def get_extremes(contour):
		'''
		Function to get the extreme points of a given contour

		Args:
		contour: contour of an image

		Returns:
		extreme_left: left-most point of the contour
		extreme_right: right-most oint of the contour
		extreme_top: top-most point of contour
		extreme_bottom: bottom-most point of contour
		'''

		extreme_left = tuple( contour[ contour[:, :, 0].argmin() ][0])
		extreme_right = tuple( contour[ contour[:, :, 0].argmax() ][0])
		extreme_top = tuple( contour[ contour[:, :, 1].argmin() ][0])
		extreme_bottom = tuple( contour[ contour[:, :, 1].argmax() ][0])

		return extreme_left, extreme_right, extreme_top, extreme_bottom


def draw_extremes(image, extreme_left, extreme_right, 
					  extreme_top, extreme_bottom, radius= 6,
					  color_left= (0, 0, 255), color_right= (0, 255, 0), 
					  color_top= (255, 0, 0), color_bottom= (255, 255, 0)):
		'''
		Funtion to draw extreme points of contours on the image

		Args:
		image: image on which points are to be drawn
		extreme_left: left-most point
		extreme_right: right-most point
		extreme_top: top-most point
		extreme_bottom: bottom-most point
		color_left: color of left-most extreme
		color_right: color of left-most extreme
		color_top: color of left-most extreme
		color_bottom: color of left-most extreme

		Returns:
		None
		'''

		cv2.circle(image, extreme_left, radius, color_left, -1)
		cv2.circle(image, extreme_right, radius, color_right, -1)
		cv2.circle(image, extreme_top, radius, color_top, -1)
		cv2.circle(image, extreme_bottom, radius, color_bottom, -1)


def count(thresholded, segmented):
		'''
		Function to count number of fingers of a hand

		Args:
		thresholded: thresholded image
		segmented: contour of a hand

		Returns:
		convex_hull: convex hull of the given segmented contour
		fingers: no. of fingers detected
		'''
		convex_hull = cv2.convexHull(segmented)
		cv2.drawContours(thresholded, [convex_hull], -1, (0, 255, 0), 2)

		extremes = get_extremes(convex_hull)

		extreme_left = extremes[0]
		extreme_right = extremes[1]
		extreme_top = extremes[2]
		extreme_bottom = extremes[3]

		# find the center of the palm
		cX = int((extreme_left[0] + extreme_right[0]) / 2)
		cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

		distance = pairwise.euclidean_distances([(cX, cY)], Y = [extreme_left, extreme_right, extreme_top, extreme_bottom])
		distance = np.squeeze(distance)
		max_distance = distance[distance.argmax()]

		#Radius of circle with 70% of max euclidean distance
		radius = int(0.7*max_distance)
		#Find circumference of the circle
		circumference = (2*np.pi*radius)

		#Take circular roi which contains hand and fingers
		circular_roi = np.zeros(thresholded.shape[:2], dtype= 'uint8')
		#Draw circular roi
		cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

		# take bit-wise AND between thresholded hand using the circular ROI as the mask
	    # which gives the cuts obtained using mask on the thresholded hand image
		circular_roi = cv2.bitwise_and(thresholded, thresholded, mask= circular_roi)

		#Find contours in circular roi
		cnts = cv2.findContours(circular_roi.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		cnts = imutils.grab_contours(cnts)

		fingers = 0

		for c in cnts:
			#Compute bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			#Increment the count of fingers only if:
			#1. The no. of points along the contour does not exceed
			#   25% of the circumference of roi
			#2. The contour region is not the wrist(bottom area)
			if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
				fingers += 1

		return convex_hull, fingers