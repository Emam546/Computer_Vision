import sys,os
sys.path.append(os.path.dirname(__file__))
from Gaussian_filter import _gaussian
import numpy as np
import cv2
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 70, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    def createProgressBar(prefix = '', suffix = '', decimals = 1, length = 70, fill = '█', printEnd = "\r",total=100):
        def printProgressBar (iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        return printProgressBar
    printProgressBar=createProgressBar(prefix, suffix, decimals, length , fill, printEnd,total )
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()
def resize_img(src, width: int | None = None, height: int | None = None, percent=None, inter: int = cv2.INTER_AREA):
    if percent != None:
        return cv2.resize(src, (0, 0), None, percent, percent, interpolation=inter)
    (h, w) = src.shape[:2]
    if width is None and height is None:
        return src
    elif width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else :
        raise Exception("Width is not defined")
    resized = cv2.resize(src, dim, interpolation=inter)
    return resized

def _bilateralFilter(src, d, sigmaColor, sigmaSpace,  borderType):
    pad = (d-1)//2
    h, w = src.shape[:2]
    new_img = np.zeros_like(src)
    src = cv2.copyMakeBorder(src, pad, pad, pad, pad,
                             borderType).astype(np.float16)
    for cy, y in progressBar(enumerate(range(pad, h+pad))):
        for cx, x in enumerate(range(pad, w+pad)):
            weight = 0
            rank = 0
            for j in range(y-pad, y+pad+1):
                for i in range(x-pad, x+pad+1):
                    gs = _gaussian(x-i, y-j, sigma=sigmaSpace)
                    gc = _gaussian(src[y, x]-src[j, i], sigma=sigmaColor)
                    val = gs*gc
                    rank += val*src[j, i]
                    weight += val
            new_img[cy, cx] = rank/weight
    return new_img


def bilateralFilter(src, d, sigmaColor, sigmaSpace,  borderType):
    pad = (d-1)//2
    h, w = src.shape[:2]
    new_img = np.zeros_like(src, "uint8")
    src = cv2.copyMakeBorder(src, pad, pad, pad, pad,
                             borderType).astype(np.float16)
    ax = np.linspace(-(d - 1) / 2., (d - 1) / 2., d)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigmaSpace))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)
    for cy, y in enumerate(progressBar(range(pad, h+pad))):
        for cx, x in enumerate(range(pad, w+pad)):
            M = src[y-pad:y+pad+1, x-pad:x+pad+1]
            diff_gaussian = np.exp(-np.absolute(M -
                                   src[y, x])**2/(2*(sigmaColor**2)))
            diff_gaussian = diff_gaussian/np.sum(diff_gaussian)
            weights = kernel*diff_gaussian
            new_img[cy, cx] = np.sum(weights*M)/np.sum(weights)
    return new_img


def _main():
    os.chdir(os.path.dirname(__file__))
    sigma_color, sigma_spitial, d = 20, 20, 7
    img = cv2.imread("../../images/Desert.jpg", 0)
    img = resize_img(img, height=300)
    blured = bilateralFilter(
        img, d, sigma_color, sigma_spitial, cv2.BORDER_REPLICATE)
    blured_2 = cv2.bilateralFilter(
        img, d, sigma_color, sigma_spitial, cv2.BORDER_REPLICATE)
    cv2.imshow("BILATERAL", blured)
    cv2.imshow("ORG_IMAGE", img)
    cv2.imshow("BILATERAL_OPENCV", blured_2)
    cv2.imshow("DIFF", cv2.threshold(cv2.absdiff(blured, img), 0, 255, 0)[1])
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
