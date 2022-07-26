from datetime import time

import numpy as np
import cv2
import os
import math
from pycv2.img.utils import ARE_EQUALE
_path=os.path.join(os.path.dirname(__file__),"Sample_images/messi5.jpg")
img=cv2.imread(_path)
AVERAGE_M=np.ones((3,3))/9

def convolve(img,kernal=AVERAGE_M,border_type=cv2.BORDER_REPLICATE):
    output_img=np.zeros(img.shape,"uint8")
    pady,padx=kernal.shape
    pady,padx=math.floor(pady/2),math.floor(padx/2)
    
    img=cv2.copyMakeBorder(img,pady,pady,padx,padx,border_type,)
    h,w,colors=img.shape
    
    for c in range(colors):
        for i in range(padx,w-padx-1):
            for j in range(pady,h-pady-1):
                M = img[j - pady:j + pady + 1, i - padx:i + padx + 1,c]
                output_img[j,i,c]=(M*kernal).sum()
    
    return output_img
def average_filter(src,size=3):
    kernal=np.ones((size,size),np.float)/(size*size)
  
    return convolve(src,kernal,)

blured_image=average_filter(img)

filter_blured_image=cv2.filter2D(img,cv2.CV_8U,AVERAGE_M)
print(ARE_EQUALE(blured_image,filter_blured_image))
cv2.imshow("Blur convlolve",blured_image)
cv2.imshow("Blur filter_2d",filter_blured_image)

cv2.imshow("original image",img)
cv2.waitKey(0)