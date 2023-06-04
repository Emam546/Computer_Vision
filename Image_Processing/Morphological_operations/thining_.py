import cv2
import os
import numpy as np


# Structuring Element
def thinning(src, ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    # Create an empty output image to hold values
    thin = np.zeros_like(src)
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(src) != 0):
        # Erosion
        erode = cv2.erode(src, kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        src = erode.copy()
    return thin


def __test1():

    img = np.zeros((100, 400), dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'TheAILearner', (5, 70), font, 2, (255), 5, cv2.LINE_AA)
    thin = thinning(img, 3)
    cv2.imshow('original', img)
    cv2.imshow('thinned', thin)
    cv2.waitKey(0)


def __test_2():
    os.chdir(os.path.dirname(__file__))
    img = cv2.imread("j.png", 0)
    thin = thinning(img, 3)
    cv2.imshow('original', img)
    cv2.imshow('thinned', thin)
    cv2.waitKey(0)


if __name__ == "__main__":
    __test_2()
