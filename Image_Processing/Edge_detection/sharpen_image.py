import cv2
import numpy as np

def sharp_edges(src,k=2,d=7,sigma=-1):
    src=src.copy()
    blured_image=cv2.blur(src,(d,d))
    M=cv2.addWeighted(src,1 ,blured_image,-1,0)
    return cv2.addWeighted(src,1,M,1*k,0,None,cv2.CV_8U)
    # Python: cv.AddWeighted(src1, alpha, src2, beta, gamma, dst) → None
    # Parameters:	
    # src1 – first input array.
    # alpha – weight of the first array elements.
    # src2 – second input array of the same size and channel number as src1.
    # beta – weight of the second array elements.
    # dst – output array that has the same size and number of channels as the input arrays.
    # gamma – scalar added to each sum.
    # dtype – optional depth of the output array; when both input arrays have the same depth, dtype can be set to -1, which will be equivalent to src1.depth()
def sharp_edges(src,k=1,d=7,sigma=-1):
    src=src.copy()
    blured_image=cv2.blur(src,(d,d))
    return cv2.addWeighted(src,1+k,blured_image,-1*k,0,None,cv2.CV_8U)
def sharpe_Laplacian(src):
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return np.absolute(cv2.filter2D(src,-1,kernel)).astype("uint8")
    

def __test():
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg")
    #blured_img=cv2.blur(img,(7,7))
    window_name="SHARPEN_IMAGE"
    def sharpend_image(val):
        sharpend_image=sharp_edges(img,val/10,d=7)
        cv2.imshow(window_name,sharpend_image)
        cv2.imshow("ABS_DIFF",cv2.absdiff(sharpend_image,img))
    cv2.namedWindow(window_name)
    cv2.createTrackbar("SCALER",window_name,10,200,sharpend_image)
    
    
    sharpend_image(1)
    cv2.imshow("BLURED IMAGE",img)
    cv2.waitKey(0)
def __test_2():
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg")
    #img=cv2.blur(img,(7,7))
    sharpend_image=sharpe_Laplacian(img)
    cv2.imshow("BLURED IMAGE",img)
    cv2.imshow("Sharped IMAGE",sharpend_image)
    cv2.waitKey(0)
def __test_3():
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg")
    img=cv2.blur(img,(7,7))
    sharpend_image=sharpe_Laplacian(img)
    cv2.imshow("BLURED IMAGE",img)
    cv2.imshow("Sharped IMAGE",sharpend_image)
    cv2.waitKey(0)
if __name__=="__main__":
    __test()