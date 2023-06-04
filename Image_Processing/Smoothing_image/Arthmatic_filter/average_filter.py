from datetime import time

import numpy as np
import cv2
import os
import math
def ARE_EQUAL(img1,img2):
    if img1 is None or img2 is None:return False
    if img1.shape==img2.shape:
        return not np.any(cv2.absdiff(img1,img2))
    else:
        return False
_path = os.path.join(os.path.dirname(__file__), "./images/messi5.jpg")
img = cv2.imread(_path)
AVERAGE_M = np.ones((3, 3))/9


def convolve(img, kernel=AVERAGE_M, border_type=cv2.BORDER_REPLICATE):
    output_img = np.zeros(img.shape, "uint8")
    pady, padx = kernel.shape
    pady, padx = math.floor(pady/2), math.floor(padx/2)

    img = cv2.copyMakeBorder(img, pady, pady, padx, padx, border_type,)
    h, w, colors = img.shape

    for c in range(colors):
        for i in range(padx, w-padx-1):
            for j in range(pady, h-pady-1):
                M = img[j - pady:j + pady + 1, i - padx:i + padx + 1, c]
                output_img[j, i, c] = (M*kernel).sum()

    return output_img


def average_filter(src, size=3):
    kernel = np.ones((size, size), np.float32)/(size*size)

    return convolve(src, kernel,)


blurred_image = average_filter(img)

filter_blured_image = cv2.filter2D(img, cv2.CV_8U, AVERAGE_M)
print(ARE_EQUAL(blurred_image, filter_blured_image))
cv2.imshow("Blur convolve", blurred_image)
cv2.imshow("Blur filter_2d", filter_blured_image)

cv2.imshow("original image", img)
cv2.waitKey(0)
