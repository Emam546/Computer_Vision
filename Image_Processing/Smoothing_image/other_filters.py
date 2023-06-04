import cv2
import numpy as np
def convolve(src,k,f,borderType):
    pad=k-1//2
    h,w=src.shape[:2]
    new_img=np.zeros_like(src)
    src=cv2.copyMakeBorder(src,pad,pad,pad,pad,borderType)
    for oy,y in enumerate(range(pad,h+pad)):
        for ox,x in enumerate(range(pad,w+pad)):
            new_img[oy,ox]=f(src[y-pad:y+pad+1,x-pad:x+pad+1])
    return new_img
def median_filter(src,k,borderType):
    def sort_(arr):
        return np.sort(arr,axis=None)[((k*k)+1)//2]
    return convolve(src,k,sort_,borderType)
def min_filter_min(src,k,borderType):
    return convolve(src,k,np.min,borderType)
def min_filter_max(src,k,borderType):
    return convolve(src,k,np.max,borderType)
