import numpy as np
import cv2
from math import floor
def resize_nearest(src,xratio,yratio):
    h,w,c=src.shape
    img=np.zeros((floor(h*yratio),floor(w*xratio),c),"uint8")
    for y in range(h):    
        for x in range(w):
            for yi in range(floor(y*yratio),floor((y+1)*yratio)):
                for xi in range(floor(x*xratio),floor((x+1)*xratio)):
                    img[yi,xi]=src[y,x]
    return img

if __name__=="__main__":
    img=cv2.imread("G:\Computer_Vision\Image_Processing\Sample_images\messi5.jpg")
    result=resize_nearest(img,0.5,0.5)
    cv2.imshow("ORG IMAGE",img)
    print(result[-1,-1])
    cv2.imshow("RESIZED IMAGE",result)
    cv2.waitKey(0)