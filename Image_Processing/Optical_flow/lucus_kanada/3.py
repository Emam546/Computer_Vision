from scipy import signal
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
def LK_OpticalFlow(Image1, Image2, kernel_size):
    I1 = np.array(Image1)
    I2 = np.array(Image2)
    S = np.shape(I1)
    
    # Apply Gaussian filter of size - 3x3
    I1_smooth = cv2.GaussianBlur(I1, kernel_size, 0)
    I2_smooth = cv2.GaussianBlur(I2, kernel_size, 0)
    
    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same')
    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same')
    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
    
    features = cv2.goodFeaturesToTrack(I1_smooth, 10000, 0.01, 5)
    feature = np.int0(features)
    plt.figure(figsize=(12,8))
    for i in feature:
        x,y = i.ravel()
        cv2.circle(I1_smooth, (x,y), 3, 0, -1)
    u = v = np.nan*np.ones(S)
    for l in feature:
        j,i = l.ravel()
        IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]])
        IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]])
        IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]])
        
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK)) # A transpose
        LK = np.array(np.matrix.transpose(LK))
        
        A1 = np.dot(LK_T, LK) # Pseudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2, LK_T)
        (u[i,j],v[i,j]) = np.dot(A3, IT)
    
    