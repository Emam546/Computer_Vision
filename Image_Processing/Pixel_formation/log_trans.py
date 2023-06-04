import cv2
import numpy as np
from math import log2
import os


def log_trans(src):
    return ((255/log2(1+np.max(src)))*np.log(src+1)).astype("uint8")


def negative_trans(src):
    return 255-src


def gamma_trans(src, gamma=0.2):
    (255/log2(1+np.max(src)))*(src.astype(np.float16)**gamma)


def _main():
    os.chdir(os.path.dirname(__file__))
    dirname = "../../images"
    for path in os.listdir(dirname):
        img = cv2.imread(os.path.join(dirname, path))
        cv2.imshow("STRETCHED IMAGE", log_trans(img))
        cv2.imshow("ORG IMAGE", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    _main()
