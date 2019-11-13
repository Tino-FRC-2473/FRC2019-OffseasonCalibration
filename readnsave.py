import numpy as np
import cv2

c = cv2.VideoCapture(0)
ret, frame = c.read()
vw = cv2.VideoWriter("supervid.mp4", -1, 30, (640, 480))

try:
	while(True):
		ret, frame = c.read()
		vw.write(frame)
except KeyboardInterrupt as e:
	print("Video Saved!")

c.release()