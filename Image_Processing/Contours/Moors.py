import matplotlib.pyplot as plt
import numpy as np


OFFSET_TABEL = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
def moors_method(src):
    contour = np.zeros(src.shape)
    starting_pixel = np.where(src>0)
    B_x = starting_pixel[0][0]#x and y direction
    B_y = starting_pixel[1][0]
    contour[B_y,B_x]=255
    id=1
    _break=False
    while not _break:
        for _ in range(9):
            if(id == 7):
                id = 0       
            c_x=B_x+OFFSET_TABEL[id+1][0]
            c_y=B_y+OFFSET_TABEL[id+1][1]        
            if (img[c_y][c_x]!=0):
                B_x,B_y = c_x,c_y                      
                if contour[B_y,B_x]:
                    _break =True
                contour[c_y,c_x]=255
                break       
            id += 1
        raise "UN DEFIEND BORDER"
    
    return contour

if __name__=="__main__":
    import cv2
    cv2.findContours()
    chirag = cv2.imread('chirag.jfif')
    gray_img = cv2.cvtColor(chirag, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(gray_img, 225,255, cv2.THRESH_BINARY_INV)
    img = cv2.dilate(img, (5,5))
    cv2.imshow('Image', chirag)
    cv2.imshow('Image_binary', img)
    #cv2.imshow('Image_contour', contour)
    cv2.waitKey(0)