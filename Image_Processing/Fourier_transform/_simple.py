import cv2
from cv2 import copyMakeBorder 
import numpy as np

cv2.namedWindow("Back_IMAGE")
def fourier_viewer(src):
    #dft = np.fft.fft2(src)
    #dft=cv2.dft(np.float32(src),flags=cv2.DFT_REAL_OUTPUT)
    dft = np.fft.fft2(img)
    fshift = np.fft.fftshift(dft)

    real_part=np.abs(fshift)
    #real_part=cv2.magnitude(fshift[:,:,0],fshift[:,:,1])
    magnitude_spectrum =20 * np.log(real_part+1)
    cv2.imshow("Fourier_image",magnitude_spectrum.astype(np.uint8))
    return dft

def ideal_filter_fourier_trans(src,r = 10):
    rows, cols = src.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((src.shape), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
    mask[mask_area] = 0

    fshift = np.fft.fftshift(src)
    return np.fft.ifftshift(fshift*mask)

def inverse_fourier_trans(src):
    fshift = np.fft.fftshift(src)
    real_part=np.abs(fshift)
    #real_part=cv2.magnitude(fshift[:,:,0],fshift[:,:,1])
    magnitude_spectrum =20 * np.log(real_part+1)
    cv2.imshow("MASKED_FOURIER",magnitude_spectrum.astype("uint8"))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    #img_back = cv2.idft(f_ishift)
    #img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    img_back = np.real(img_back).astype("uint8")
    cv2.imshow("Back_IMAGE",img_back)

def GaussianHighfilter(src,r=10):
    rows, cols = src.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2, )
    fshift = np.fft.fftshift(src)
    expon=np.exp(-(mask_area**2)/2*(r**2))
    
    return  np.fft.ifftshift(expon*fshift)
def GausssianLowFilter(src,sigma=10):
    rows, cols = src.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2, )
    fshift = np.fft.fftshift(src)
    
    expon=np.exp(-(mask_area**2)/2*(sigma**2))
    expon=1-expon
    return  np.fft.ifftshift(expon*fshift)
def GaussianHighfilter(src,sigma=10):
    rows=max(src.shape[:2])
    
    kernal=cv2.getGaussianKernel(rows,sigma)
    
    kernal=np.dot(kernal,kernal.T)
    #kernal=copyMakeBorder(kernal,abs(rows-row)//2,(rows-column)//2,(rows-row)//2,(rows-column)//2,cv2.BORDER_CONSTANT,value=0)
    row,column=src.shape[:2]
    kernal=kernal[rows//2-row//2:rows//2+row//2,rows//2-column//2:rows//2+column//2,]
    
    kernal=np.fft.fft2(kernal)
    kernal=np.fft.fftshift(kernal)
    fshift = np.fft.fftshift(src)
    
    return  np.fft.ifftshift(fshift*kernal)
if __name__=="__main__":
    img=cv2.imread(r"D:\Learning\Learning_python\opencv\images\messi5.jpg",0)
    
    def fourier_trans(radius):
        dimg=fourier_viewer(img)
        dimg=GaussianHighfilter(dimg,radius/100)
        inverse_fourier_trans(dimg)
    cv2.createTrackbar("Radius","Back_IMAGE",0,1000,fourier_trans)
    fourier_trans(10)
    cv2.waitKey(0)