import cv2
import numpy as np
import os
_path=os.path.join(os.path.dirname(__file__),"Sample_images/sudoku-original.jpg")
img=cv2.imread(_path,0)
equl_hist=cv2.equalizeHist(img)
cv2.imshow("org img",img)
cv2.imshow("hist",equl_hist)
cv2.waitKey(0)