import cv2
import os
window_name = "GAUSSIAN FILTER"

# cv2.createTrackbar("size",GAUSSIAN_FILTER,0,30,lambda val:gaussian(cv2.getTrackbarPos("sigma",GAUSSIAN_FILTER),val))

_current_img = None


def gaussian(val):
    if (_current_img is None):
        return
    gaussian_img = cv2.GaussianBlur(_current_img, (0, 0), val)
    gaussian_img_2 = cv2.GaussianBlur(_current_img, (0, 0), val**2)
    cv2.imshow(window_name, gaussian_img)
    laplacian_img = cv2.Laplacian(gaussian_img, cv2.CV_64F)
    laplacian_img_2 = cv2.Laplacian(gaussian_img_2, cv2.CV_64F)
    # diff_lablace=cv2.absdiff(lablacien_img,lablacien_img_2)
    cv2.imshow("LABLACIEN_IMAGE", laplacian_img)


def laplacian_gaussian(src):
    global _current_img
    _current_img = src.copy()
    cv2.setTrackbarPos("sigma", window_name, 1)
    # cv2.setTrackbarPos("size",GAUSSIAN_FILTER,1)
    gaussian(3)


cv2.namedWindow(window_name)
cv2.createTrackbar("sigma", window_name, 0, 255, gaussian)


def _main():
    os.chdir(os.path.dirname(__file__))
    img = cv2.imread("./sweets.png")
    laplacian_gaussian(img)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
