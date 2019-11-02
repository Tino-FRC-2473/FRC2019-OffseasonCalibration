# add imports
import math
import cv2
import numpy as np
import subprocess
import imghdr
import traceback
import os
from matplotlib import pyplot as plt
import random

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
# print(true_green_vals)


low_green = np.array([69,152,0])
high_green= np.array([86,219,255])



def get_new_hsv(res):

    

    global true_green_vals
    print(len(true_green_vals))
    for i in range(10):
        row=random.randrange(0,len(res))
        # print(res[row])
        true_green_vals = np.append(true_green_vals, np.reshape(np.array(res[row]), (1, 1, 3)), 0)

        


    h=true_green_vals[:,:,0]
    s=true_green_vals[:,:,1]
    v=true_green_vals[:,:,2]

    plt.figure()
    hue_hist=plt.hist(h,180,[0,179]); plt.show()


    print()

    low_h, low_s, low_v = (h.mean() - 2*h.std()), (s.mean() - 1*s.std()), (v.mean() - 2.5*v.std())
    high_h, high_s, high_v = (h.mean() + 2*h.std()), (s.mean() + 1*s.std()), (v.mean() + 2.5*v.std())

    return np.array([int(low_h), int(low_s), int(low_h)]), np.array([int(high_h), int(high_s), int(high_v)])
    

while(True):

    #start of cv code
    # frame = cv2.imread('test_photos/0degrees_24inches.png')

    _, frame = vid.read()


    # cv2.imshow('camera', frame)

    small = cv2.pyrDown(frame)

    hsv_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # print(hsv[...,2])
    dark = hsv[...,2] < 32
    # print(dark)
    hsv[dark] = 0
    h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    h = np.clip(h*0.005*hist_scale, 0, 1)


    vis = hsv_map*h[:,:,np.newaxis] / 255.0


    h_vals = vis[:,:, 0]
    # print(255*h_vals[h_vals != 0])
    h_vals = 255*h_vals[h_vals != 0]
    hue_hist=plt.hist(h_vals,180,[0,180])
  


    

    

    # low_green = np.array([int(h_vals.mean() - (0.3*h_vals.std())), 181,0])
    # high_green= np.array([int(h_vals.mean() + (0.1*h_vals.std())),255,255])

    mask = cv2.inRange(hsv_2, low_green, high_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    green_vals = np.array([])
    
    res=res[~(res==0).all(2)]

    low_green, high_green = get_new_hsv(res)

    print("low green: ", low_green)
    print("high green: ", high_green)
    

    # print("Res:")
    # print(res)
    # print(true_green_vals)
    
            

    
    
            
    #sorts contours by area 
    #contours.sort(key = cv2.contourArea, reverse = True)

    contours.sort(key = lambda countour: cv2.boundingRect(countour)[0])

    rotated_boxes = []
    rotated_rect1 = None
    rotated_rect2 = None
    rotated_rect3 = None
    pinX = 0
    pinY = 0

    total_contour_area = 0

    res = cv2.bitwise_and(frame,frame, mask= mask)

    for c in contours:
        #ignore anything below hatch panel level
        centery = cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]/2
        if True:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            _, _, rot_angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)


        total_contour_area += cv2.contourArea(c)

    # if it detects one rectangle, draw center of rectangle
    if(len(rotated_boxes)  == 1):
        rotated_rect = rotated_boxes[0]
        rect_x = rotated_rect.get_center().x
        rect_y = rotated_rect.get_center().y
        cv2.circle(frame, (rect_x, rect_y), 1, (255, 0, 0), thickness=5)

        cv2.drawContours(frame, [rotated_rect.box], 0, (0, 0, 255), 2)
        cv2.imshow("Contours", mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Res", res)
        cv2.waitKey(1)

    # if there are less than two rectangles, return -99, -1
    if (len(rotated_boxes) < 2):
        cv2.imshow("Contours", mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Res", res)
        cv2.waitKey(1)
        #return -99, -1

    cv2.imshow("Contours", mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("Res", res)
    # cv2.waitKey(1)

