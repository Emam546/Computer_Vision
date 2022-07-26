import numpy as np
import cv2
import math
import time
def rotated_mask(img,ksize=3,border=cv2.BORDER_REPLICATE):
    print("START ALGORITHM")
    start=time.time()
    h,w=img.shape[:2]
    img=cv2.copyMakeBorder(img,ksize,ksize,ksize,ksize,border)
    output_image=np.zeros_like(img)
    pad=math.floor(ksize/2)
    kernal=np.ones((ksize,ksize))/(ksize*ksize)
    for c in range(ksize,w):
        for r in range(ksize,h):
            sigma,averages=[],[]
            for kr in range(pad*-1,pad+1):
                for kc in range(pad*-1,pad+1):
                    filt=int((
                        img[
                            r-pad+kr:r+pad+kr+1,
                            c-pad+kc:c+pad+kc+1]*kernal).sum())
                    varience=sum([abs(img[r+ir+kr,c+ic+kc]-filt) for ir in range(pad*-1,pad+1) for ic in range(pad*-1,pad+1)])/(ksize*ksize)
                        
                    averages.append(filt)
                    sigma.append(varience)
            
            index=np.argmin(np.array(sigma))
            output_image[r,c]=averages[index]
    print("TIME TAKEN :{:.2f}".format(time.time()-start))
    return output_image[ksize:h,ksize:w]
def _main():
    orgimg=cv2.imread("messi5.jpg",0)
    cv2.imshow("FILTEDRED IMAGE",rotated_mask(orgimg))
    cv2.imshow("ORG IMAGE",orgimg)
    cv2.waitKey(0)
if __name__=="__main__":
    _main()
    

    