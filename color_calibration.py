import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('FRC2019-OffseasonVisionAlignment/test_photos/0degrees_18inches.png')
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img)

h=h.ravel()
s=s.ravel()
v=v.ravel()

hue_hist=plt.hist(h,180,[0,180])
sat_hist=plt.hist(s,256,[0,256])
val_hist=plt.hist(v,256,[0,256])

print("Hue: " + str(h.mean()))
print("Saturation: " + str(s.mean()))
print("Value: " + str(v.mean()))
