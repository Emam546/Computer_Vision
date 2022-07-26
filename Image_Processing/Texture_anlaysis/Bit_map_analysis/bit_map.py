import numpy as np

import cv2
_WIDTH_BIT=8
# Read the image in greyscale
def binary_represntor(src):
    return np.vectorize(np.binary_repr)(src, width=_WIDTH_BIT)
def seperate_binary(src):
    return [src.view((str,1)).reshape((src.shape[0]*src.shape[1]),-1)[:,x].reshape(src.shape).astype("uint8")*255 for x in range(_WIDTH_BIT) ]
def _main():
    img = cv2.imread('bit1.jpg',0)
    binary_img=binary_represntor(img).astype(np.str)
    images=seperate_binary(binary_img)
    finalr = cv2.hconcat(images[:4])
    finalv =cv2.hconcat(images[4:])
    
    # Vertically concatenate
    final = cv2.vconcat([finalr,finalv])
    
    # Display the images
    cv2.imshow('a',final)
    cv2.waitKey(0) 
if __name__=="__main__":
    _main()
