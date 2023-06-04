import math
import cv2
import numpy as np
_COLORS = [0, 0, 0]
HIST_SIZE = 256
NAMES = ["BLUE CHANNEL", "GREEN CHANNEL", "RED CHANNEL"]
HIST_H, HIST_W = 100, 500


def display_histograme(bgr_planes: list, names=NAMES):
    histograms = [cv2.calcHist(bgr_planes, [ix], None, [
                               HIST_SIZE], [0, 255], accumulate=False) for ix in range(len(bgr_planes))]
    bin_w = int(round(HIST_W/HIST_SIZE))

    for i_h, (windowname, hist) in enumerate(zip(names, histograms)):
        histImage = np.zeros((HIST_H, HIST_W, 3), "uint8")
        cv2.normalize(hist, hist, alpha=0, beta=HIST_H,
                      norm_type=cv2.NORM_MINMAX)

        for i in range(1, HIST_SIZE):
            color = _COLORS.copy()
            color[i_h] = 255
            pos = (bin_w*(i-1), HIST_H - int(hist[i-1])), (bin_w*(i), HIST_H)
            cv2.rectangle(histImage, *pos, color, thickness=-1)
        yield histImage
        cv2.imshow(windowname, histImage)

    raise StopIteration


def thresh_finder(src, thresh, delta_T=1.0):
    x_low, y_low = np.where(src <= thresh)
    x_high, y_high = np.where(src > thresh)
    mean_low = np.mean(src[x_low, y_low])
    mean_high = np.mean(src[x_high, y_high])
    new_thresh = (mean_low + mean_high)/2
    # Step-4: Calculate the new threshold
    if abs(new_thresh-thresh) < delta_T:
        return new_thresh
    else:
        return thresh_finder(src, new_thresh, delta_T)


def balanced_hist_threshing(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e)//2
    # Left side weight
    w_l = np.sum(b[0][0:i_m+1])
    # Right side weight
    w_r = np.sum(b[0][i_m+1:i_e+1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) >= i_m:
                w_l += b[0][i_m+1]
                w_r -= b[0][i_m+1]
                i_m += 1
    return i_m


def otsus_method(hist):
    hist.reshape(-1).astype(np.float16)
    cols = np.arange(hist.size, dtype=np.float16)

    tot_pix = np.sum(hist)
    variances = []
    for thresh in range(1, hist.size-1):
        bg_pix = np.sum(hist[:thresh])
        fg_pix = np.sum(hist[thresh:])
        Wb = bg_pix/tot_pix
        Wf = fg_pix/tot_pix
        Ub = np.sum(cols[:thresh]*hist[:thresh])/bg_pix
        Uf = np.sum(cols[thresh:]*hist[thresh:])/fg_pix
        var = Wb*Wf*((Ub-Uf)**2)
        variances.append(var)
    print(variances)
    variances = np.nan_to_num(variances, False)
    return np.argmax(variances)+1


def otsus_2_method(hist):
    hist.reshape(-1).astype(np.float16)
    cols = np.arange(hist.size, dtype=np.float16)
    tot_pix = np.sum(hist)
    variances = []
    for thresh in range(1, hist.size-1):
        bg_pix = np.sum(hist[:thresh])
        fg_pix = np.sum(hist[thresh:])
        Wb = bg_pix/tot_pix
        Wf = fg_pix/tot_pix
        Ub = np.sum(cols[:thresh]*hist[:thresh])/bg_pix
        Uf = np.sum(cols[thresh:]*hist[thresh:])/fg_pix

        Qb = np.sum(((cols[:thresh]-Ub)**2)*hist[:thresh])/bg_pix
        Qf = np.sum(((cols[thresh:]-Uf)**2)*hist[thresh:])/fg_pix
        var = Wb*(Qb**2)+Wf*(Qf**2)
        variances.append(var)

    variances = np.nan_to_num(variances, False, np.inf)
    return np.argmin(variances)+1


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
    else:
        raise Exception("Width is not defined")
    resized = cv2.resize(src, dim, interpolation=inter)
    return resized


def __test():
    import time
    img = cv2.imread("./images/aerial_image.jpg", 0)
    img = resize_img(img, 500)
    display_histograme([img])
    hist = cv2.calcHist([img], [0], None, [HIST_SIZE], [0, HIST_SIZE-1])
    print(hist.size)
    print("start")
    start = time.time()
    thresh = balanced_hist_threshing(hist)
    print("end time", time.time()-start)
    thresh_img = cv2.threshold(img, thresh, 255, 0)[1]
    cv2.imshow("THRESH", thresh_img)
    cv2.imshow("IMAGE", img)
    cv2.waitKey(0)


def __test_otsus_method():
    img = cv2.imread("./images/Desert.jpg", 0)
    img = resize_img(img, 500)
    hist = cv2.calcHist([img], [0], None, [HIST_SIZE], [0, HIST_SIZE-1])

    thresh = otsus_method(hist)
    thresh_2 = otsus_2_method(hist)

    print(thresh, thresh_2)
    thresh_img = cv2.threshold(img, thresh, 255, 0)[1]
    cv2.imshow("THRESH", thresh_img)
    cv2.imshow("IMAGE", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    __test_otsus_method()
