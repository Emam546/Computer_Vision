import numpy as np
import cv2

def min_max_streching(src):
    src=src.copy()
    min_value=np.min(src)
    return (255*((src.astype(np.float)-min_value)/(np.max(src)-min_value))).astype("uint8")
if __name__=="__main__":

    img=cv2.imread(r"D:\Learning\Computer_Vision\Image_Processing\Contrast_streching\stretch_original.jpg")
    result=min_max_streching(img)
    cv2.imshow("ORG_IMAGE",img)
    cv2.imshow("Min_Max",result)
    cv2.waitKey(0)