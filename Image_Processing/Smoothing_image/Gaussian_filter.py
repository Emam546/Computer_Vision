import math
import numpy as np
from pycv2.tools import progressBar
import cv2
def _gaussian(*weights,sigma):
    x1=2*math.pi*(sigma**2)
    var=-sum((val**2 for val in weights))
    x2=np.exp(var/(2*(sigma**2)))
    return x2/x1
    

def getGaussianKernel(d, sigma):
    pad=(d-1)//2 
    kernel=np.zeros((d,d),np.float64)
    for j,vj in enumerate(range(-pad,pad+1)):
        for i,vi in enumerate(range(-pad,pad+1)):
            kernel[j,i]=_gaussian(vj,vi,sigma=sigma)
    return kernel/np.sum(kernel)
def Gaussian_blur(src,d,sigma,borderType):
    
    kernel=getGaussianKernel(d,sigma)
    return cv2.filter2D(src,-1,kernel,borderType=borderType)
    pad=(d-1)//2 
    h,w=src.shape[:2]
    new_img=np.zeros_like(src)
    #src=cv2.copyMakeBorder(src,pad,pad,pad,pad,borderType).astype(np.float)
    # for y in progressBar(range(pad,h)):
    #     for x in range(pad,w):
    #         new_img[y,x]=np.sum(src[y-pad:y+pad+1,x-pad:x+pad+1]*kernel)
    return new_img
def getGaussianKernel(d=5, sigma=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(d - 1) / 2., (d - 1) / 2., d)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)/100
    return kernel / np.sum(kernel)
def _main():
    sigma=1
    size=3
    kernel=getGaussianKernel(size,sigma)
    kernal2=cv2.getGaussianKernel(size,sigma)
    kernal2=np.dot(kernal2,kernal2.T)
    print(kernal2,kernel,sep="\n")
    print(np.sum(kernel),np.sum(kernal2))
if __name__=="__main__":
    _main()
    
    # img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg")
    # blured=cv2.filter2D(img,-1,kernel)
    # blured_2=cv2.filter2D(img,-1,kernal2)
    
    # cv2.imshow("BLURED_IMAGE",blured)
    # cv2.imshow("BLURED_IMAGE_2",blured_2)
    # cv2.imshow("ORG_IMAGE",img)
    # cv2.imshow("DIFF",cv2.threshold(cv2.absdiff(blured,blured_2),0,255,0)[1])
    # cv2.waitKey(0)