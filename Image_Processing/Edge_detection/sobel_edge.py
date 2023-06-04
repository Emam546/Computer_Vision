import cv2
import numpy as np
import os
os.chdir(os.path.dirname(__file__))
img = cv2.imread("../../images/messi5.jpg", cv2.IMREAD_GRAYSCALE)
WINDOW_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])
WINDOW_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

sobelx64f = cv2.filter2D(img, cv2.CV_64F, WINDOW_X)
sobely64f = cv2.filter2D(img, cv2.CV_64F, WINDOW_Y)
magnitude = np.sqrt((sobelx64f**2+sobely64f**2))
viewmag = np.absolute(magnitude)
viewmag = np.uint8(viewmag)


cv2.imshow("sobel x", sobelx64f)
cv2.imshow("sobel y", sobely64f)
cv2.imshow("magnituide", viewmag)
cv2.waitKey(0)
