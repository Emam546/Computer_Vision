import numpy as np
import cv2
def __simple_frist_derative():
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg",0)
    xd=np.array([-1,0,1])
    x_derative=cv2.filter2D(img,cv2.CV_64F,xd,)
    absolute=np.absolute(x_derative)
    derative_8u=absolute.astype("uint8")
    cv2.imshow("x_derative_frist",derative_8u)
    cv2.imshow("x_derative_64_frist",x_derative)
def _simple_second_derative():
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg",0)
    xd=np.array([-1,2,-1])
    x_derative=cv2.filter2D(img,cv2.CV_64F,xd,)
    absolute=np.absolute(x_derative)
    derative_8u=absolute.astype("uint8")
    cv2.imshow("x_derative_second",derative_8u)
    cv2.imshow("x_derative_64_second",x_derative)
    
if __name__=="__main__":
    __simple_frist_derative()
    _simple_second_derative()
    cv2.waitKey(0)