import cv2
import numpy as np

def iterative_thresh(img):
    h,w=img.shape
    frist_pixels=img[0,0],img[h-1,0],img[0,w-1],img[h-1,w-1]
    bg=sum(frist_pixels)/4
    ob=(sum(img.reshape(-1))-sum(frist_pixels))/((h*w)-4)
    last_tresh=None

    while True:
        final_tresh=int((bg+ob)/2)
        if final_tresh==last_tresh:
            return final_tresh
        last_tresh=final_tresh
        mask=cv2.threshold(img,final_tresh,255,0)[1]
        
        #foreground pixles
        ob_pix=img[mask==255].reshape(-1)
        ob=np.sum(ob_pix)/len(ob_pix)

        bg_pix=img[mask==0].reshape(-1)
        bg=np.sum(bg_pix)/len(bg_pix)
        

if __name__=="__main__":
    import os
    folder="G:\Learning_python\opencv\images"
    for img in os.listdir(folder):
        if os.path.splitext(img)[1] in [".jpg",".png"]:
            orgimg=cv2.imread(os.path.join(folder,img),0)
            thresh=iterative_thresh(orgimg)
            print(img,thresh)
            cv2.imshow("FILTEDRED IMAGE",cv2.threshold(orgimg,thresh,255,0)[1])
            cv2.imshow("ORG IMAGE",orgimg)
            cv2.waitKey(0)
