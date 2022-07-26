
import numpy as np
import cv2
from pycv2.tools import progressBar

import sys,os
sys.path.append(os.path.dirname(__file__))
from Gaussian_filter import _gaussian,getGaussianKernel

def bilateralFilter(src, d,sigmaColor, sigmaSpace,  borderType):
    pad=(d-1)//2
    h,w=src.shape[:2]
    new_img=np.zeros_like(src)
    src=cv2.copyMakeBorder(src,pad,pad,pad,pad,borderType).astype(np.float)
    for cy,y in progressBar(enumerate(range(pad,h+pad))):
        for cx,x in enumerate(range(pad,w+pad)):
            weight=0
            rank=0
            for j in range(y-pad,y+pad+1):
                for i in range(x-pad,x+pad+1):
                    gs=_gaussian(x-i,y-j,sigma=sigmaSpace)
                    gc=_gaussian(src[y,x]-src[j,i],sigma=sigmaColor)
                    val=gs*gc
                    rank+=val*src[j,i]
                    weight+=val
            new_img[cy,cx]=rank/weight
    return new_img
def bilateralFilter(src, d,sigmaColor, sigmaSpace,  borderType):
    pad=(d-1)//2
    h,w=src.shape[:2]
    new_img=np.zeros_like(src,"uint8")
    src=cv2.copyMakeBorder(src,pad,pad,pad,pad,borderType).astype(np.float)
    ax = np.linspace(-(d - 1) / 2., (d - 1) / 2., d)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigmaSpace))
    kernel = np.outer(gauss, gauss)
    kernel/=np.sum(kernel)
    for cy,y in enumerate(progressBar(range(pad,h+pad))):
        for cx,x in enumerate(range(pad,w+pad)):
            M=src[y-pad:y+pad+1,x-pad:x+pad+1]
            diff_gaussian=np.exp(-np.absolute(M-src[y,x])**2/(2*(sigmaColor**2)))
            diff_gaussian=diff_gaussian/np.sum(diff_gaussian)
            weights=kernel*diff_gaussian
            new_img[cy,cx]=np.sum(weights*M)/np.sum(weights)
    return new_img

if __name__=="__main__":
    from pycv2.img.utils import resizeimage_keeprespective
    sigma_color,sigma_spitial,d=20,20,7
    img= cv2.imread("D:\Learning\Learning_python\opencv\images\Desert.jpg",0)
    img=resizeimage_keeprespective(img,height=300)
    blured=bilateralFilter(img,d,sigma_color,sigma_spitial,cv2.BORDER_REPLICATE)
    blured_2=cv2.bilateralFilter(img,d,sigma_color,sigma_spitial,cv2.BORDER_REPLICATE)
    cv2.imshow("BILATERRAL",blured)
    cv2.imshow("ORG_IMAGE",img)
    cv2.imshow("BILATERRAL_OPENCV",blured_2)
    cv2.imshow("DIFF",cv2.threshold(cv2.absdiff(blured,img),0,255,0)[1])
    cv2.waitKey(0)
