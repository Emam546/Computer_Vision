import cv2
import numpy as np
import os
HIST_SIZE = 256


def Mode_hist(img, minimum_peak=20):
    hist = cv2.calcHist([img], [0], None, [HIST_SIZE],
                        [0, 255], accumulate=False)

    hist = [(i, val) for i, val in enumerate(hist.reshape(-1))]
    hist.sort(key=lambda x: x[1], reverse=True)
    bigest_val = hist[0][0]
    for val, _ in hist:
        if bigest_val-val > minimum_peak:
            return cv2.threshold(img, (bigest_val+val)//2, 255, 0)
    return cv2.threshold(img, (bigest_val+hist[-1][0])//2, 255, 0)


def Multi_hist(img, minimum_peak=20):
    hist = cv2.calcHist([img], [0], None, [HIST_SIZE],
                        [0, 255], accumulate=False)

    hist = [(i, val) for i, val in enumerate(hist.reshape(-1))]
    hist.sort(key=lambda x: x[1], reverse=True)
    bigest_val = hist[0][0]
    for val, _ in hist:
        if bigest_val-val > minimum_peak:
            return hyst_thresh(img, bigest_val, val)
    return hyst_thresh(img, bigest_val, hist[-1][0])


def hyst_thresh(edge_img: np.ndarray, high_thresh: float = 200, low_thresh: float = 100):

    matrix = np.zeros(edge_img.shape, dtype=np.uint8)  # create an empty matrix of type uint8

    # find positions of all elements that are above low threshold
    r, c = np.where(edge_img > low_thresh)

    matrix[r, c] = 255  # set elements above low threshold to 255 (white)

    # find positions of all elements that are above high threshold
    r, c = np.where(edge_img >= high_thresh)


    # this gives me label which is number of groups(labels of groups) and a matrix neighbors
    label, neighbours = cv2.connectedComponents(matrix)

    # Example:

    # label = 4(group 0, group 1 group 2 group 3)
    #
    # neighbors
    #
    # 0 0 0 1 0
    # 0 1 1 1 0
    # 0 0 0 0 0
    # 2 2 2 0 3

    for i in range(1, label):
        y, z = np.where(neighbours == i)
        k1 = np.isin(r, y)
        k2 = np.isin(c, z)
        if not any(k1*k2):
            matrix[y, z] = 0
    return matrix


def _main():
    os.chdir(os.path.dirname(__file__))
    folder = "../../../images"
    for img in os.listdir(folder):
        if os.path.splitext(img)[1] in [".jpg", ".png"]:
            orgimg = cv2.imread(os.path.join(folder, img), 0)
            cv2.imshow("FILTERED IMAGE", Multi_hist(orgimg))
            cv2.imshow("ORG IMAGE", orgimg)
            cv2.waitKey(0)


if __name__ == "__main__":
    _main()
