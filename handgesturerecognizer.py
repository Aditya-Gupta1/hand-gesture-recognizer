import cv2
import imutils
import numpy as np
from contours import get_max_contour
from contours import get_extremes
from contours import draw_extremes
import contours

class HandGestureRecognizer:

	def __init__(self, width= 800):
		'''
		Initializes the window to be processed
		
		Args:
		width: width of the output video

		Returns:
		None
		'''
		self.bg = None #Background
		self.width = width

	def display_frame_rate(self, frame, frame_counter, limit, position):
		'''
		Function to display frame counter

		Args:
		frame: current frame in video
		frame_counter: current frame number
		limit: no. of frames required for background identification

		Returns:
		None
		'''
		color = (0, 0, 255)
		if frame_counter >= limit:
			color = (0, 255, 0)
		cv2.putText(frame, 'Frame '+str(frame_counter), position, 
					cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		
	def running_average(self, frame, weight= 0.5):
		'''
		Function to calculate the running average to identify background

		Args:
		frame: current frame of video feed
		weight: to compute weighted avergae

		Returns:
		None
		'''
		if self.bg is None:
			self.bg = frame.copy().astype('float')
			return

		cv2.accumulateWeighted(frame, self.bg, weight)


	def start(self, video_source= 0, rectangle_color= (255, 0, 0),
			  rectangle_thickness= 2, break_key= 'q', frame_rate= False,
			  limit= 30, weight = 0.5, display_thresholded= False):
		'''
		Main function to start recognizing hand gestures

		Args:
		video_source: path to video source. To put in cv2.VideoCapture method
		rectangle_color: color of rectangular frame
		rectangle_thickness: thickness of rectangular frame
		break_key: key to be pressed to stop video feed
		frame_rate: to display frame rate in the output video
		limit: no. of frames required for background identification
		weight: weight to calculate weighted average of frame and background
		display_thresholded: to display thresholded image or not

		Returns:
		Video output feed showing hand gestures.
		'''
		if self.width < 500:
			print("Width can't be less than 500px. Please Try Again")
			return


		self.left = int(0.59375 * self.width)
		self.right = int(0.96875 * self.width)
		self.top = int(0.03125 * self.width)
		self.bottom = int(0.4125 * self.width)

		cam = cv2.VideoCapture(video_source)
		
		frame_counter = 0

		while True:

			_, frame= cam.read()

			#Increment frame counter
			frame_counter += 1
			#Flip to eliminate mirror effect
			frame = cv2.flip(frame, 1)
			#Resizing by maintaining the aspect ratio
			frame = imutils.resize(frame, width= self.width)
			#Calculate regio of interest to look for hand
			region_of_interest = frame[self.top: self.bottom , self.left: self.right]

			#Caluate running average to identify background correctly
			if frame_counter <= limit:
				self.running_average(region_of_interest, weight)
			else:
			
				#Find contour of hand
				contour = get_max_contour(region_of_interest, self.bg)
				#If contour present
				if contour is not None:

					thresholded, max_contour = contour

					#Get extreme points of a contour
					extremes = get_extremes(max_contour)

					#Calculate center of hand and draw it
					cX = int((extremes[0][0] + extremes[1][0]) / 2)
					cY = int((extremes[2][1] + extremes[3][1]) / 2)
					cv2.circle(region_of_interest, (cX, cY), 6, (255, 0, 0), -1)

					#Count no. of fingers
					convex_hull, no_of_fingers= contours.count(thresholded, max_contour)

					#Print fingers count
					finger_pos = (int(0.03125 * self.width), int(0.0625 * self.width))
					cv2.putText(frame, "Fingers : "+str(no_of_fingers), finger_pos,
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

					#Draw convex hull
					cv2.drawContours(region_of_interest, [convex_hull], -1, (0, 255, 255), 2)

					#Display thresholded image if opted by the user
					if display_thresholded:
						cv2.imshow("thresholded", thresholded)

					#Draw contour of a hand
					cv2.drawContours(region_of_interest, 
									 [max_contour], -1, (0, 255, 0), 2)

					#Draw contour extremes
					draw_extremes(region_of_interest,
								  extremes[0], extremes[1], extremes[2], extremes[3])


			#Draw window frame of region of interest
			cv2.rectangle(frame, (self.left,self.top), (self.right, self.bottom),
						  rectangle_color, rectangle_thickness)
			
			#Display frame counter is opted
			if frame_rate:

				frame_rate_pos = (int(0.03125*self.width), int(0.125*self.width))

				self.display_frame_rate(frame, frame_counter, limit, frame_rate_pos)

			if frame_counter >= limit-5 and frame_counter <= limit+5:

				counter_position = (int(0.03125 * self.width), int(0.450 * self.width))

				cv2.putText(frame, "Background Analysis Complete", counter_position,
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

			cv2.imshow('Video', frame)

			if cv2.waitKey(10)&0xFF == ord(break_key):
				break

		cam.release()
		cv2.destroyAllWindows()
