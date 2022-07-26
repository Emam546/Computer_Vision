import numpy as np
import cv2
from pycv2.img.utils import *
from math import floor
def resize_bilinear(src,xratio,yratio):
    h,w,c=src.shape
    src=cv2.copyMakeBorder(src,0,1,0,1,cv2.BORDER_REPLICATE).astype(np.float)
    #print(h,w,*src.shape[:2])
    img=np.zeros((floor(h*yratio),floor(w*xratio),c),np.float)
    #calculating first row values
    for x in range(w):
        y2=floor(1*yratio)
        x1,x2=floor(x*xratio),floor((x+1)*xratio)
        for (rx1,xi),rx2 in zip(enumerate(range(x1,x2)),range(x2-x1,0,-1)):
            img[y2,xi]=(src[0,x]*rx2+src[0,x+1]*rx1)/(x2-x1)

    for y in range(h-1):    
        for x in range(w):
            y1,y2=floor(y*yratio),floor((y+1)*yratio)
            x1,x2=floor(x*xratio),floor((x+1)*xratio)
            for (rx1,xi),rx2 in zip(enumerate(range(x1,x2)),range(x2-x1,0,-1)):
                img[y2,xi]=(src[y+1,x]*rx2+src[y+1,x+1]*rx1)/(x2-x1)
            
            for (ry1,yi),ry2 in zip(enumerate(range(y1,y2)),range(y2-y1,0,-1)) :
                for xi in range(x1,x2):
                    img[yi,xi]=(img[y1,x1]*ry2+img[y2,x1]*ry1)/(y2-y1)
    

    return img.astype("uint8")
def __main():
    img=cv2.imread("messi5.jpg")
    result=resize_bilinear(img,2,2)
    h,w=result.shape[:2]
    org_img=cv2.resize(img,(w,h))
    cv2.imshow("ORG IMAGE",org_img)
    print(result[-1,-1],org_img[-1,-1])
    cv2.imshow("RESIZED IMAGE",result)
    cv2.waitKey(0)
if __name__=="__main__":
    __main()
