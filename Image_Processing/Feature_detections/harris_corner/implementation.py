import cv2 
import numpy as np
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
def harris_corner(src,ksize=3 ,k=0.04,borderType=cv2.BORDER_REPLICATE):
    sobel_x=cv2.getDerivKernels(0,1,ksize)
    sobel_x=np.dot(sobel_x[0],sobel_x[1].reshape(1,ksize))
    sobel_y=cv2.getDerivKernels(1,0,ksize)
    sobel_y=np.dot(sobel_y[0],sobel_y[1].reshape(1,ksize))

    sobelxImage = cv2.filter2D(src,cv2.CV_64F,sobel_x,borderType=borderType)
    sobelyImage = cv2.filter2D(src,cv2.CV_64F,sobel_y,borderType=borderType)
    

    #sobelGrad=np.sqrt(gx**2+gy**2)
    sobelxyImage = sobelxImage* sobelyImage
    sobelxImage,sobelyImage=sobelxImage**2,sobelyImage**2

    sobelxImage = cv2.GaussianBlur(sobelxImage,(ksize,ksize),-1)
    sobelyImage = cv2.GaussianBlur(sobelyImage,(ksize,ksize),-1)
    sobelxyImage = cv2.GaussianBlur(sobelxyImage,(ksize,ksize),-1)

    det = sobelxImage *sobelyImage - (sobelxyImage ** 2)
    trace = k * ((sobelxImage +sobelyImage) ** 2)

    #finding the harris response
    return det - trace
if __name__=="__main__":
    import os
    #os.chdir(os.path.dirname(__file__))
    img=cv2.imread("arrowimage.jpg")
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gaussian_blur=cv2.GaussianBlur(gray_img,(3,3),-1)
    dst=harris_corner(gray_img)
    viewmag = np.absolute(dst)
    viewmag = np.uint8(viewmag)
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>dst.max()*0.01]=[0,0,255]
    cv2.imshow("COR IMAGE",img)
    
    cv2.imshow("HARRIS",viewmag)
    cv2.waitKey(0)