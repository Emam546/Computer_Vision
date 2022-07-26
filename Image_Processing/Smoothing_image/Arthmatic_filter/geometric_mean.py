import cv2
import numpy as np
def geometric_mean(src,ksize,borderType=cv2.BORDER_REPLICATE):
    pad=(ksize-1)//2
    h,w=src.shape
    new_img=np.zeros_like(src,np.float)
    pad_src=cv2.copyMakeBorder(src,pad,pad,pad,pad,borderType)
    return(new_img**(1/(ksize**2))).astype(src.dtype)
def harmonic_mean(src,ksize,borderType=cv2.BORDER_REPLICATE):
    kernel=np.ones((ksize,ksize))
    res=src.copy().astype(np.float)
    new_img=cv2.filter2D(1/res,cv2.CV_64F,kernel,borderType)
    return ((ksize**2)/new_img).astype(src.dtype)
def contraHarmonic(src,ksize,Q,borderType=cv2.BORDER_REPLICATE):
    kernel=np.ones((ksize,ksize))
    res=src.copy().astype(np.float)
    sum1=cv2.filter2D(res**(Q+1),cv2.CV_64F,kernel,borderType)
    sum2=cv2.filter2D(res**Q,cv2.CV_64F,kernel,borderType)
    return (sum1/sum2).astype(src.dtype)


def __test():
    img=cv2.imread("D:\Learning\Learning_python\opencv\images\messi5.jpg",0)
    window_name="HORMONIC"
    def sharpend_image(val=None):
        if val%2==0:return
        sharpend_image=geometric_mean(img,val,1)
        cv2.imshow(window_name,sharpend_image)
        cv2.imshow("ABS_DIFF",cv2.absdiff(sharpend_image,img))
    cv2.namedWindow(window_name)
    sharpend_image(3)
    cv2.createTrackbar("SCALER",window_name,3,20,sharpend_image)
    cv2.waitKey(0)
    
if __name__=="__main__":
   __test()
    