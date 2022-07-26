import cv2
import numpy as np
def calcBackProject(src,roi):
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    
    M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    R = M/I
    h,s,v = cv2.split(hsvt)

    B = R[h.ravel(),s.ravel()]
    B = np.minimum(B,1)
    B = B.reshape(hsvt.shape[:2])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(B,-1,disc,B)
    B = np.uint8(B)
    cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)