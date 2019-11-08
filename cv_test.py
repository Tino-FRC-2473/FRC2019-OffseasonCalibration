import math
import cv2
import numpy as np
import subprocess
import imghdr
import traceback
import os
from matplotlib import pyplot as plt
import random
import pdb

vid = cv2.VideoCapture('video2.mp4')

hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:,:,0] = h
hsv_map[:,:,1] = s
hsv_map[:,:,2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
hist_scale = 10

rgb_data = np.loadtxt('green_data.csv', dtype= np.uint8, delimiter=',')
bgr_data = np.copy(rgb_data)
bgr_data[:,0] = rgb_data[:,2]
bgr_data[:,2] = rgb_data[:,0]


bgr_data = np.reshape(bgr_data, (79, 1, 3))

true_green_vals = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2HSV)

pixel_count=len(true_green_vals)


low_green = np.array([69,152,0])
high_green= np.array([86,219,255])

def get_new_hsv(res):]

    global true_green_vals
    global low_green
    global high_green
    
    if(res is None):
        return low_green, high_green
    

    for i in range(100):
        row=random.randrange(0,len(res))
        true_green_vals = np.append(true_green_vals, np.reshape(np.array(res[row]), (1, 1, 3)), 0)

        
    h=true_green_vals[:,:,0]
    s=true_green_vals[:,:,1]
    v=true_green_vals[:,:,2]

    plt.figure()


    low_h, low_s, low_v = (h.mean() - 3.25*h.std()), (s.mean() - 3.5*s.std()), (v.mean() - 3.25*v.std())
    high_h, high_s, high_v = (h.mean() + 3.25*h.std()), (s.mean() + 3.5*s.std()), (v.mean() + 3.25*v.std())

    return np.array([int(low_h), int(low_s), int(low_h)]), np.array([int(high_h), int(high_s), int(high_v)])
    

while(True):
    # frame = cv2.imread('test_photos/0degrees_24inches.png')

    _, frame = vid.read()

    # cv2.imshow('camera', frame)

    small = cv2.pyrDown(frame)

    hsv_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
  
    mask = cv2.inRange(hsv_2, low_green, high_green)
    
    greens=hsv_2[np.where((mask==255))]

    low_green, high_green = get_new_hsv(greens)

    print("low green: ", low_green)
    print("high green: ", high_green)
    
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("True greens", greens)
    cv2.waitKey(1)
