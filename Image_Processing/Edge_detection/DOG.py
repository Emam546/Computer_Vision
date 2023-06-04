import os,sys
os.chdir(os.path.dirname(__file__))
sys.path.append("../")
import cv2
from Smoothing_image.Gaussian_filter import getGaussianKernel_native
name_window = "DIFF_IMAGE"
cv2.namedWindow(name_window)


def __test():
    size, sigma1, sigma2 = 7, 1, 50
    cv2.createTrackbar("Size", name_window, size, 9, lambda size: change_trackbar(
        cv2.getTrackbarPos("Sigma", name_window), size))
    cv2.createTrackbar("Sigma", name_window, 0, 50, lambda sigma: change_trackbar(
        sigma, cv2.getTrackbarPos("Size", name_window)))

    def change_trackbar(sigma, size):
        if size % 2 == 0 and sigma > 0:
            return
        # print(sigma,size)

        kernel_1 = getGaussianKernel_native(3, sigma=sigma)
        kernel_2 = getGaussianKernel_native(size, sigma=1)
        img1 = cv2.filter2D(src, -1, kernel_1)
        # img2=cv2.filter2D(src,-1,kernel_2)

        diff = cv2.absdiff(img1, src)
        diff[diff > 0] = 255
        cv2.imshow(name_window, diff)
        cv2.imshow("IMAGE", src)

        cv2.waitKey(0)

    # img_path=os.path.join(Images_path,"")
    src = cv2.imread("../../images/messi5.jpg", 0)
    change_trackbar(cv2.getTrackbarPos("Sigma", name_window),
                    cv2.getTrackbarPos("Size", name_window))


def __test_2():
    Images_path = "../../images"
    for name in os.listdir(Images_path):
        img_path = os.path.join(Images_path, name)
        src = cv2.imread(img_path, 0)
        img1 = cv2.GaussianBlur(src, (3, 3), 0)
        img2 = cv2.GaussianBlur(src, (5, 5), 0)

        diff = cv2.absdiff(img1, img2)
        # diff[diff>100]=255
        cv2.imshow("DOG_IMAGE", diff)
        cv2.imshow("IMAGE", src)
        cv2.waitKey(0)


if __name__ == "__main__":
    __test()
