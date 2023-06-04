import cv2
import numpy as np
import os


def open(img, shape=(5, 5)):
    eroded = cv2.erode(img, shape)
    return cv2.dilate(eroded, shape)


def ARE_EQUAL(img1, img2):
    if img1 is None or img2 is None:
        return False
    if img1.shape == img2.shape:
        return not np.any(cv2.absdiff(img1, img2))
    else:
        return False


def _main():
    os.chdir(os.path.dirname(__file__))
    img = cv2.imread("j.png", 0)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (5, 5))
    my_opened_img = open(img)
    print(ARE_EQUAL(opened_img, my_opened_img))
    cv2.imshow("MY OPENED IMAGE", my_opened_img)
    cv2.imshow("ORG OPENED IMG", opened_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
