import math
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

def _gaussian(*weights, sigma):
    x1 = 2*math.pi*(sigma**2)
    var = -sum((val**2 for val in weights))
    x2 = np.exp(var/(2*(sigma**2)))
    return x2/x1


def getGaussianKernel_native(d, sigma):
    pad = (d-1)//2
    kernel = np.zeros((d, d), np.float64)
    for j, vj in enumerate(range(-pad, pad+1)):
        for i, vi in enumerate(range(-pad, pad+1)):
            kernel[j, i] = _gaussian(vj, vi, sigma=sigma)
    return kernel/np.sum(kernel)


def Gaussian_blur(src, d, sigma, borderType):

    kernel = getGaussianKernel(d, sigma)
    return cv2.filter2D(src, -1, kernel, borderType=borderType)


def Gaussian_blur_native(src, d, sigma, borderType):
    kernel = getGaussianKernel(d, sigma)

    pad = (d-1)//2
    h, w = src.shape[:2]
    new_img = np.zeros_like(src)
    src = cv2.copyMakeBorder(src, pad, pad, pad, pad,
                             borderType).astype(np.float32)
    for y in progressBar(range(pad, h)):
        for x in range(pad, w):
            new_img[y, x] = np.sum(src[y-pad:y+pad+1, x-pad:x+pad+1]*kernel)
    return new_img


def getGaussianKernel(d=5, sigma=1.):
    # creates gaussian kernel with side length `l` and a sigma of `sig`

    ax = np.linspace(-(d - 1) / 2., (d - 1) / 2., d)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)/100
    return kernel / np.sum(kernel)


def _main():
    sigma = 1
    size = 3
    kernel = getGaussianKernel_native(size, sigma)
    kernel2 = cv2.getGaussianKernel(size, sigma)
    kernel2 = np.dot(kernel2, kernel2.T)
    print(kernel2, kernel, sep="\n")
    print(np.sum(kernel), np.sum(kernel2))


if __name__ == "__main__":
    _main()
