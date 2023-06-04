import cv2
import os
os.chdir(os.path.dirname(__file__))
img = cv2.imread("../../images/sudoku-original.jpg", 0)
equalize_hist = cv2.equalizeHist(img)
cv2.imshow("org img", img)
cv2.imshow("hist", equalize_hist)
cv2.waitKey(0)
