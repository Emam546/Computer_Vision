import cv2
import numpy as np
from skimage import color
img = cv2.imread('coins1.jpg')
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = 1)

dist = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
ret, dist = cv2.threshold(dist, 0.6*dist.max(), 255, 0)
dist=dist.astype(np.uint8)
cv2.imshow("SEGMENTED",dist)
markers = cv2.circle(dist, (15,15), 5,255, -1)
_, markers = cv2.connectedComponents(dist)
markers = markers+10

markers[dist==0] = 0
print(np.unique(markers))

markers = cv2.watershed(img, markers)
img2 = color.label2rgb(markers, bg_label=0)
img2[markers == -1] = [0,255,255]
cv2.imshow('Colored Grains', img2)  
cv2.imshow('Overlay on original image', img)
cv2.waitKey(0)