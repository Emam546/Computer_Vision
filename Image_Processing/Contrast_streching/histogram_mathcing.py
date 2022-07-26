import cv2
import numpy as np
import math
HIST_SIZE=256
def adabtive_hist_equ(src,kernal=(8,8)):
    pady,padx=math.floor(kernal[1]/2),math.floor(kernal[0]/2)
    new_image=cv2.copyMakeBorder(src,pady,padx,pady,padx,cv2.BORDER_REPLICATE)
    un_defiend_image=np.zeros_like(src,"uint8")
    h,w=new_image.shape
    total_region=padx*pady*4
    for iy,y in enumerate(range(pady,h-pady)):
        for ix,x in enumerate(range(padx,w-padx)):
            rank=0
            for j in range(y-pady,y+pady):
                for i in range(x-padx,x+padx):
                    if new_image[y,x]>=new_image[j,i]:
                        rank+=1
            un_defiend_image[iy,ix]=round(255*(rank/total_region))
            #print(rank,x)
   # print(total_region)
    return un_defiend_image
def equlization(src):
    hist=cv2.calcHist([src],[0],None,[HIST_SIZE],(0,255),accumulate = False)
    total_num=src.shape[1]*src.shape[0]
    return [round(255*sum([int(hist[i][0])/total_num for i in range(256)]))]
def find_nearest_above(hist,target):
        diff = hist - target
        mask = np.ma.less_equal(diff, -1)
        # We need to mask the negative differences
        # since we are looking for values above
        if np.all(mask):
            c = np.abs(diff).argmin()
            return c # returns min index of the nearest if target is greater than any value
        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()
def match_hist(hist1,hist2):
    b=[]
    for data in hist1[:]:
        b.append(find_nearest_above(hist2,data))
    return np.array(b,dtype='uint8')
def hist_match(original, specified): 
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')

    return b[bin_idx].reshape(oldshape)
if __name__=="__main__":
    img=cv2.imread(r"D:\Learning\Computer_Vision\Image_Processing\Contrast_streching\stretch_original.jpg",0)  
    cv2.imshow("ORG IMAGE",img)
    equalized_image=cv2.equalizeHist(img)
    cv2.imshow("EQUALIZED IMAGE",equalized_image)
    adaptive_hist=adabtive_hist_equ(img,(60,60))
    cv2.imshow("ADAPTIVE EQUALIZED IMAGE",adaptive_hist)
    cv2.waitKey(0)