import cv2
import numpy as np
def convert_water_art(src,d_kernal,sigma_color=0,sigma_space=0):
    blured_img=cv2.bilateralFilter(src,d_kernal,sigma_color,sigma_space,)
    return blured_img