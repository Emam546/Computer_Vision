import cv2
import numpy as np

def sharp_edges(src,k=2,d=7,sigma=-1):
    src=src.copy()
    blured_image=cv2.blur(src,(d,d))
    M=cv2.addWeighted(src,1 ,blured_image,-1,0)
    return cv2.addWeighted(src,1,M,1*k,0,None,cv2.CV_8U)
def sharp_edges(src,k=1,d=7,sigma=-1):
    src=src.copy()
    blurred_image=cv2.blur(src,(d,d))
    return cv2.addWeighted(src,1+k,blurred_image,-1*k,0,None,cv2.CV_8U)
def sharpe_Laplacian(src):
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return np.absolute(cv2.filter2D(src,-1,kernel)).astype("uint8")
    

def __test():
    img=cv2.imread("./images/messi5.jpg")
    #blurred_img=cv2.blur(img,(7,7))
    window_name="SHARPEN_IMAGE"
    def sharpened_image(val):
        sharpened_image=sharp_edges(img,val/10,d=7)
        cv2.imshow(window_name,sharpened_image)
        cv2.imshow("ABS_DIFF",cv2.absdiff(sharpened_image,img))
    cv2.namedWindow(window_name)
    cv2.createTrackbar("SCALER",window_name,10,200,sharpened_image)
    
    
    sharpened_image(1)
    cv2.imshow("BLURED IMAGE",img)
    cv2.waitKey(0)
def __test_2():
    img=cv2.imread("./images/messi5.jpg")
    #img=cv2.blur(img,(7,7))
    sharpened_image=sharpe_Laplacian(img)
    cv2.imshow("BLURED IMAGE",img)
    cv2.imshow("Sharped IMAGE",sharpened_image)
    cv2.waitKey(0)
def __test_3():
    img=cv2.imread("./images/messi5.jpg")
    img=cv2.blur(img,(7,7))
    sharpened_image=sharpe_Laplacian(img)
    cv2.imshow("BLURED IMAGE",img)
    cv2.imshow("Sharped IMAGE",sharpened_image)
    cv2.waitKey(0)
if __name__=="__main__":
    __test()