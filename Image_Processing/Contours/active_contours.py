
import cv2
from cv2 import BORDER_CONSTANT
from cv2 import EVENT_MOUSEMOVE
import numpy as np
from pycv2.img.utils import all_closetest_nodes,distance
# from pykeyboard import keyboard
# from pykeyboard.keys import ENTER
DRAWING_WINDOW = "Drawing_window"

def _draw_circls(src,points,line=False):
    if src.ndim==2:
        src=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    _drawing =src.copy()
    for i,(y,x) in enumerate(points):

        cv2.circle(_drawing, (x,y), 5, (255, 0, 0), -1)
        if line:
            cv2.line(_drawing,points[i-1][::-1],points[i][::-1],(0, 255, 0),1)

    cv2.imshow(DRAWING_WINDOW, _drawing)

def _put_points(src,lines=False):
    points = []
    state=[False]    
    def draw_circle(event, x, y, flags, param):
        y,x=min(src.shape[0],max(0,y)),min(src.shape[1],max(0,x))
        if event == cv2.EVENT_LBUTTONDOWN:
            state[0]=True
            points.append((y, x))
        elif event == cv2.EVENT_LBUTTONUP:
            state[0]=False
        elif event == cv2.EVENT_RBUTTONDOWN:
            state[0]=False
            if len(points)>0:
                _points,pos=np.array(points),np.array((y,x))
                dist=np.sqrt(np.sum((_points-pos)**2,axis=1))
                for i in np.where(dist>20):
                    points.pop(i)
        elif event==cv2.EVENT_MOUSEMOVE and state[0]:
            if distance(points[-1],(y, x))>20:
                points.append((y, x))

        _draw_circls(src,points,lines)
    cv2.namedWindow(DRAWING_WINDOW)
    cv2.setMouseCallback(DRAWING_WINDOW, draw_circle)
    _draw_circls(src,points,lines)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    state[0]=False
    return points
def _gradient(src,pad=0):
    _img=cv2.GaussianBlur(src,(7,7),-1)
    sobel_x = cv2.Sobel(_img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(_img, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude=cv2.GaussianBlur(magnitude,(7,7),-1)
    return cv2.copyMakeBorder(magnitude,pad,pad,pad,pad,cv2.BORDER_CONSTANT,None,0)
    
def active_contour_Energy(src, window_size=7, thresh=0):
    points=_put_points(src)
    pad = (window_size-1)//2
    points=np.array(points)+(pad,pad)
    magnitude=_gradient(src,pad)

    diff = float("inf")
    E = 0
    while diff > thresh:
        for i, (y, x) in enumerate(points.copy()):
            M = magnitude[y-pad:y+pad+1, x-pad:x+pad+1]
            points[i] = np.array((y, x))+\
                np.array(np.unravel_index([np.argmax(M)],M.shape)).reshape(-1)-\
                np.array((pad+1, pad+1))
            _draw_circls(src,points)
            cv2.waitKey(30)
        energy = sum([magnitude[y, x] for (y, x) in points])
        diff = abs(E-energy)
        E = energy
    print(E)
    cv2.waitKey(0)

def active_contour(src, window_size=7,alpha=0.4,beta=0.5, thresh=0):
    points=_put_points(src,True)
    pad = (window_size-1)//2
    magnitude=_gradient(src,pad)
    points=np.array(points)+(pad,pad)
    rows, cols=magnitude.shape[:2]
    y,x=np.ogrid[:rows, :cols]
    y,x=np.where(magnitude!=np.NaN)
    grids=np.concatenate([np.array(y).reshape(*magnitude.shape,1),np.array(x).reshape(*magnitude.shape,1)],axis=2)

    diff=float("inf")
    E=0
    while diff > thresh:
        TOTAL_ENERGY=0
        for i, (y, x) in enumerate(points.copy()):
            M = magnitude[y-pad:y+pad+1, x-pad:x+pad+1]
            G=grids[y-pad:y+pad+1, x-pad:x+pad+1]
            max_diff=-(M-magnitude[y,x])

            ni=(i+1)%len(points) 
            pi=i-1 
            smooth=np.sum(((points[ni]-G)**2)*alpha,axis=2)
            elistic=np.sum((points[ni]-(2*G)+points[pi])*beta,axis=2)
            

            vals=max_diff+smooth+elistic
            _draw_circls(magnitude.copy().astype("uint8"),points,True)
            dir=np.array(np.unravel_index([np.argmin(vals)],M.shape)).reshape(-1)
            TOTAL_ENERGY+=vals[dir[0],dir[1]]
            points[i] = np.array((y, x))+dir-np.array((pad+1, pad+1))
            cv2.waitKey(1)
    
        diff = abs(E-TOTAL_ENERGY)
        E = TOTAL_ENERGY
    print(E)
    cv2.waitKey(0)


if __name__ == "__main__":
    img = cv2.imread(
        r"D:\Learning\Computer_Vision\Image_Processing\Contours\coin.jpg", 0)
    active_contour(img,7,1,1,thresh=20)
