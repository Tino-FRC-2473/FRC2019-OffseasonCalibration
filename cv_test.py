# add imports
import math
import cv2
import numpy as np
import subprocess
import imghdr
import traceback
import os

while(True):
    frame = cv2.imread('test_photos/20degrees_18inches.png')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_green = np.array([62,181,93])
    high_green= np.array([88,255,174])

    mask = cv2.inRange(hsv, low_green, high_green)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.waitKey(3)

    # if there are less than two rectangles, return -99, -1
    if (len(rotated_boxes) < 2):
        cv2.imshow("Contours", mask)
        cv2.imshow("Frame", frame)
        cv2.waitKey(3)
        #return -99, -1

    cv2.imshow("Contours", mask)
    cv2.imshow("Frame", frame)
    cv2.waitKey(3)
