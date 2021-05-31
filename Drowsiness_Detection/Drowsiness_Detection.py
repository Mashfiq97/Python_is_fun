from scipy.spatial import distance
from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd

def sound_alarm(path):
	# play an alarm sound                  
	playsound.playsound(path)
	
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
def amplitude(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	amp = (A + B) / 2.0
	return amp
	
thresh = 0.25
frame_check = 24

COUNTER = 0
ALARM_ON = False

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
time.sleep(1.0)
df = []
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		ampli1 = amplitude(leftEye)
		ampli2 = amplitude(rightEye)
		total_ampli = (ampli1 + ampli2) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			if (1==1):
                                df.append(
                                        {
                                                'Amplitude': total_ampli,
                                                'Left_Ear': leftEAR,
                                                'Right_Ear': rightEAR,
                                                'Ear': ear,
                                                'Eye_closed': flag,
                }
            )
			print (str(flag) +" observing "+ str(ear))
			if flag >= frame_check:
                                # if the alarm is not on, turn it on
				# check to see if an alarm file was supplied,
				# and if so, start a thread to have the alarm
				# sound played in the background
				sound_alarm("alarm.wav")
                                
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Drowsy")
				#pd.DataFrame(df)
                #df = pd.DataFrame(df)
                #df.to_csv (r'D:\AAAAAAAAAA\fileone.csv', index = False, header=True)
		else:
			flag = 0
			ALARM_ON = False
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
