import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

img = cv.imread('test_photos/0degrees_18inches.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)



h=h.ravel()
s=s.ravel()
v=v.ravel()

print("mean", hsv.mean())



hue_hist=plt.hist(h,180,[0,180])
sat_hist=plt.hist(s,256,[0,256])
val_hist=plt.hist(v,256,[0,256])

plt.show()

print("Hue: " + str(h.mean()))
print("Saturation: " + str(s.mean()))
print("Value: " + str(v.mean()))

h_low=h.mean()-h.std()
h_high=h.mean()+h.std()
s_low=s.mean()-s.std()
s_high=s.mean()+s.std()
v_low=v.mean()-v.std()
v_high=v.mean()+v.std()

print("Range for Hue: (" + str(math.floor(h_low)) + ", " + str(math.floor(h_high)) + ")")
print("Range for Saturation: (" + str(math.floor(s_low)) + ", " + str(math.floor(s_high)) + ")")
print("Range for Value: (" + str(math.floor(v_low)) + ", " + str(math.floor(v_high)) + ")")
