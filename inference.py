# Importing necessary libraries, including OpenCV for computer vision, 
# Mediapipe for pose estimation, NumPy for numerical operations, and 
# Keras for loading the pre-trained model
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Defines a function inFrame to check if specific landmarks in the pose are visible, returning a boolean.
def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False


# Loads a pre-trained model (model.h5) and associated label information (labels.npy).
model  = load_model("model.h5")
label = np.load("labels.npy")


# Sets up the Mediapipe Pose model (holis) and drawing utilities for visualization.
# initializing mediapipe pose for pose detection
holistic = mp.solutions.pose 
holis = holistic.Pose()

# for visualizing drawing landmarks
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Enters a continuous loop for real-time pose classification.
while True:
	lst = []
# Captures a frame, flips it horizontally, and processes it using the Mediapipe Pose model.
	_, frm = cap.read()

	window = np.zeros((940,940,3), dtype="uint8")

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

	frm = cv2.blur(frm, (4,4))
	# Checks if pose landmarks are detected and if the specified landmarks are visible.
	#  If conditions are met, extracts features and makes predictions using the pre-trained model. 
	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - res.pose_landmarks.landmark[0].x)
			lst.append(i.y - res.pose_landmarks.landmark[0].y)

		lst = np.array(lst).reshape(1,-1)

		p = model.predict(lst)
		pred = label[np.argmax(p)]

		if p[0][np.argmax(p)] > 0.75:
			cv2.putText(window, pred , (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)

		else:
			cv2.putText(window, "Asana is either wrong or not trained" , (100,180),cv2.FONT_ITALIC, 1.8, (0,0,255),3)

	else: 
		cv2.putText(frm, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

	# Draws pose landmarks on the video frame and displays the result along with classification information.	# 
	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
							connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
							 landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))


	window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))

	cv2.imshow("window", window)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

