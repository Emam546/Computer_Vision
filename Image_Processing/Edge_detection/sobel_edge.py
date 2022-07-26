import cv2
import numpy  as np
import os
from pycv2.img.utils import ARE_EQUALE
import math
_path=os.path.join(os.path.dirname(__file__),"Sample_images/messi5.jpg")
img=cv2.imread(_path,cv2.IMREAD_GRAYSCALE)
WINDOW_X=np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],
])
WINDOW_Y=np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

sobelx64f=cv2.filter2D(img,cv2.CV_64F,WINDOW_X)
sobely64f=cv2.filter2D(img,cv2.CV_64F,WINDOW_Y)
magnituide= np.sqrt((sobelx64f**2+sobely64f**2))
viewmag = np.absolute(magnituide)
viewmag = np.uint8(viewmag)


cv2.imshow("sobel x",sobelx64f)
cv2.imshow("sobel y",sobely64f)
cv2.imshow("magnituide",viewmag)
cv2.waitKey(0)

