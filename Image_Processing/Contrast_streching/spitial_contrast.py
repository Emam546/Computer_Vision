import numpy as np
import cv2
import os


def min_max_stretching(src):
    src = src.copy()
    min_value = np.min(src)
    return (255*((src.astype(np.float16)-min_value)/(np.max(src)-min_value))).astype("uint8")


def _main():
    os.chdir(os.path.dirname(__file__))
    img = cv2.imread("stretch_original.jpg")
    result = min_max_stretching(img)
    cv2.imshow("ORG_IMAGE", img)
    cv2.imshow("Min_Max", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
