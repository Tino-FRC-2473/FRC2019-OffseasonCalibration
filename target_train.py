from numpy import genfromtxt
import numpy as np
import cv2
from matplotlib import pyplot as plt

rgb_data = np.loadtxt('green_data.csv', dtype= np.uint8, delimiter=',')
bgr_data = np.copy(rgb_data)
bgr_data[:,0] = rgb_data[:,2]
bgr_data[:,2] = rgb_data[:,0]


bgr_data = np.reshape(bgr_data, (79, 1, 3))

hsv = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2HSV)


plt.figure()
hue_hist=plt.hist(hsv[:, :, 0],180,[0,179]); plt.show()
plt.figure()
sat_hist=plt.hist(hsv[:, :, 1],256,[0,255]); plt.show()
plt.figure()
val_hist=plt.hist(hsv[:, :, 2],256,[0,255]); plt.show()
print(hsv[:, :, 0].std())
low_h, low_s, low_v = (hsv[:, :, 0].mean() - 2.5*hsv[:, :, 0].std()), (hsv[:, :, 1].mean() - 2.5*hsv[:, :, 1].std()), (hsv[:, :, 2].mean() - 2.5*hsv[:, :, 2].std())
high_h, high_s, high_v = (hsv[:, :, 0].mean() + 2.5*hsv[:, :, 0].std()), (hsv[:, :, 1].mean() + 2.5*hsv[:, :, 1].std()), (hsv[:, :, 2].mean() + 2.5*hsv[:, :, 2].std())


print([low_h, low_s, low_v])
print([high_h, high_s, high_v])


# print(hsv)
# frame = cv2.imread('test_photos/0degrees_18inches.png')
# print(frame)
import pdb
pdb.set_trace()

