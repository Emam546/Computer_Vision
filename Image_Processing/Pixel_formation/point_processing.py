import numpy as np
import cv2
import os


def iter_img(img, fucnt):
    h, w, ch = img.shape
    newimg = np.zeros(img.shape, "uint8")
    for x in range(ch):
        for r in range(h):
            for c in range(w):
                newimg[r, c, x] = fucnt(img[r, c, x])
    return newimg


def increase_contrast(value: int):
    return min(255, value*2)


def lighten(value):
    return min(255, int(value)+128)


def darken(value: int):
    return min(255, value/2)


def invert(value: int):
    return max(0, 255-int(value))


def _main():
    os.chdir(os.path.dirname(__file__))
    img = cv2.imread("../../images/messi5.jpg")
    cv2.imshow("image contrast", iter_img(img, lighten))
    cv2.imshow("image", iter_img(img, increase_contrast))
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
