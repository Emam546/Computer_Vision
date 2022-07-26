# Description: Detect pedestrians in a video using the
#   Histogram of Oriented Gradients (HOG) method

import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
from imutils.object_detection import non_max_suppression  # Handle overlapping

# Make sure the video file is in the same directory as your code
filename = r'D:\Learning\Learning_python\opencv\videos\Pedestrian_Walking.mp4'

cv2.namedWindow("Frame")

def main():

    # Create a HOGDescriptor object
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(filename)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def detect(val=None): 
        if val==None:
            val=cv2.getTrackbarPos("TRACK_BAR","Frame")
            
        # Detect people
        # image: a single frame from the video
        # winStride: step size in x and y direction of the sliding window
        # padding: no. of pixels in x and y direction for padding of sliding window
        # scale: Detection window size increase coefficient
        # bounding_boxes: Location of detected people
        # weights: Weight scores of detected people
        # Tweak these parameters for better results
        (bounding_boxes, weights) = hog.detectMultiScale(frame,winStride=(10, 10),padding=(4, 4),
                                                            scale=val/100)
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
        # bounding_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bounding_boxes])
        # selection = non_max_suppression(bounding_boxes,
        #                                 probs=None,
        #                                 overlapThresh=0.45)
        # for (x1, y1, x2, y2) in selection:
        #     cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.createTrackbar("TRACK_BAR","Frame",100,500,detect)
    for idx in range(0,frames_num, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break

        
        
        orig_frame=frame.copy()
        detect()
        cv2.imshow("Frame", frame)
        cv2.imshow("CORRECTED_FRAME",orig_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
