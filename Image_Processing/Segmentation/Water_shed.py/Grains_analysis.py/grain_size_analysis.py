import cv2
import numpy as np
from skimage import color
from skimage.segmentation import clear_border



pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)

def Mark_analysis(src,mask,percentage=0.2):
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    
    opening = clear_border(opening) 
    sure_bg = cv2.dilate(opening,kernel,iterations=2)
    _, sure_fg = cv2.threshold(dist_transform,percentage*dist_transform.max(),255,0)
    #Remove edge touching grains
    #Check the total regions found before and after applying this. 
    sure_fg=np.uint8(sure_fg)
    # cv2.imshow("Sure_fg",sure_fg)
    # cv2.imshow("Sure_bg",sure_bg)
    # unknown = cv2.subtract(sure_bg,sure_fg)
    # cv2.imshow("Sure_unknown", unknown)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10

    markers[sure_fg==0] = 0
    return cv2.watershed(src,markers)
    
    #Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
def __test():

    
    #os.chdir(os.path.dirname(__file__))
    \
    img = cv2.imread("grains1.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    def show(percentage):
        markers=Mark_analysis(img,thresh,percentage/10)
        print(len(np.unique(markers))-1)
        img2 = color.label2rgb(markers, bg_label=0)
        img2[markers == -1] = [0,255,255]
        cv2.imshow('Colored Grains', img2)  
    cv2.namedWindow('Colored Grains')
    cv2.createTrackbar("percent",'Colored Grains',1,10,show)
    show(1)
    cv2.imshow('Overlay on original image', img)
    cv2.waitKey(0)
    
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
def __test_2():
    img = cv2.imread("Osteosarcoma_01.tif")
    img=resize_img(img,600)

    
    lab =cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    clahe=cv2.createCLAHE(100,(64,64))
    lab[:,:,0]=cv2.equalizeHist(lab[:,:,0])
    img =cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    #gray=img[:,:,0]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img=cv2.inRange(img,(160,0,0),(260,255,255))
     #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("THRESH",thresh)
    def show(percentage):
        markers=Mark_analysis(img,thresh,percentage/10)
        print(len(np.unique(markers))-1)
        img2 = color.label2rgb(markers, bg_label=0)
        img2[markers == -1] = [0,0,0]
        cv2.imshow('Colored Grains', img2)  
    cv2.namedWindow('Colored Grains')
    cv2.createTrackbar("percent",'Colored Grains',1,10,show)
    show(1)
    cv2.imshow('Overlay on original image', img)
    cv2.waitKey(0)
def __org_code():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import ndimage
    from skimage import measure, color, io

    img = cv2.imread("grains2.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)

    #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # Morphological operations to remove small noise - opening
    #To remove holes we can use closing
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    from skimage.segmentation import clear_border
    opening = clear_border(opening) #Remove edge touching grains
    #Check the total regions found before and after applying this. 


    #Now we know that the regions at the center of cells is for sure cells
    #The region far away is background.
    #We need to extract sure regions. For that we can use erode. 
    #But we have cells touching, so erode alone will not work. 
    #To separate touching objects, the best approach would be distance transform and then thresholding.

    # let us start by identifying sure background area
    # dilating pixes a few times increases cell boundary to background. 
    # This way whatever is remaining for sure will be background. 
    #The area in between sure background and foreground is our ambiguous area. 
    #Watershed should find this area for us. 
    sure_bg = cv2.dilate(opening,kernel,iterations=2)


    # Finding sure foreground area using distance transform and thresholding
    #intensities of the points inside the foreground regions are changed to 
    #distance their respective distances from the closest 0 value (boundary).
    #https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

    #Let us threshold the dist transform by 20% its max value.
    #print(dist_transform.max()) gives about 21.9
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

    #0.2* max value seems to separate the cells well.
    #High value like 0.5 will not recognize some grain boundaries.

    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg,sure_fg)

    #Now we create a marker and label the regions inside. 
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0. 
    #For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)

    #One problem rightnow is that the entire background pixels is given value 0.
    #This means watershed considers this region as unknown.
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.

    #Now we are ready for watershed filling. 
    markers = cv2.watershed(img,markers)
    #The boundary region will be marked -1
    #https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1


    #Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
    img[markers == -1] = [0,255,255]  

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('Overlay on original image', img)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

if __name__=="__main__":
    __test_2()