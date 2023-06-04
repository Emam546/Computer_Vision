import cv2

import numpy as np
import os
_COLOS = [0, 0, 0]
HIST_SIZE = 256
NAMES = ["BLUE CHANNEL", "GREEN CHANNEL", "RED CHANNEL"]
HIST_H, HIST_W = 100, 500


def display_histogram(bgr_planes: list, names=NAMES):
    histograms = [cv2.calcHist(bgr_planes, [ix], None, [HIST_SIZE], [
                               0, 255], accumulate=False) for ix in range(len(bgr_planes))]
    bin_w = int(round(HIST_W/HIST_SIZE))

    for i_h, (windowname, hist) in enumerate(zip(names, histograms)):
        histImage = np.zeros((HIST_H, HIST_W, 3), "uint8")
        cv2.normalize(hist, hist, alpha=0, beta=HIST_H,
                      norm_type=cv2.NORM_MINMAX)

        for i in range(1, HIST_SIZE):
            color = _COLOS.copy()
            color[i_h] = 255
            pos = (bin_w*(i-1), HIST_H - int(hist[i-1])), (bin_w*(i), HIST_H)
            cv2.rectangle(histImage, *pos, color, thickness=-1)
        cv2.imshow(windowname, histImage)


def equalization(src):
    hist = cv2.calcHist([src], [0], None, [HIST_SIZE],
                        [0, 255], accumulate=False)
    output_img = src.copy()
    total_num = src.shape[1]*src.shape[0]
    for l in range(HIST_SIZE):
        output_img[src == l] = round(
            HIST_SIZE*sum([int(hist[i][0])/total_num for i in range(l+1)]))
    return output_img


def equal_hist(src):
    hist = np.zeros(HIST_SIZE,)
    s_values, s_counts = np.unique(src, return_counts=True)
    hist[s_values] = s_counts
    s_quantiles = np.cumsum(hist).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    return np.around(s_quantiles*255)


def ARE_EQUAL(img1, img2):
    if img1 is None or img2 is None:
        return False
    if img1.shape == img2.shape:
        return not np.any(cv2.absdiff(img1, img2))
    else:
        return False


def __main():
    os.chdir(os.path.dirname(__file__))
    _path = os.path.join(os.path.dirname(__file__), "../../images/messi5.jpg")
    img = cv2.imread(_path, 0)
    first_img, second_img = cv2.equalizeHist(img), equalization(img)
    display_histogram([first_img], ["EQUALIZED IMAGE"])
    display_histogram([second_img], ["MY EQUALIZED IMAGE"])
    cv2.imshow("EQUALIZE IMAGE", first_img)
    cv2.imshow("MY_EQUALIZED IMAGE", second_img)
    print("it is the same", ARE_EQUAL(first_img, second_img))
    cv2.waitKey(0)


if __name__ == "__main__":
    __main()
