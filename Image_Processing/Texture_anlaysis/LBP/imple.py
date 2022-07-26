import cv2
import numpy as np
ORDER=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
def lbp_filter(img,border_type=cv2.BORDER_REPLICATE):
    new_img=np.zeros_like(img)
    
    h,w=img.shape
    img=cv2.copyMakeBorder(img,1,1,1,1,border_type,)
    for y in range(1,h):
        for x in range(1,w):
            current_val=img[y,x]
            new_img[y,x]=sum([(2**idx) for idx,(i,j) in enumerate(ORDER) if img[i+y,j+x]>=current_val])
    return new_img
if __name__=="__main__":
    img=cv2.imread("G:\Computer_Vision\Image_Processing\Texture_anlaysis\LBP\\faces.jpg",0)
    for img in np.array_split(img, 4):
        for img in np.array_split(img, 6,axis=1):
            cv2.imshow("ORG IMAGE",img)
            cv2.imshow("LBL IMAGE",lbp_filter(img))
            cv2.waitKey(0)
    