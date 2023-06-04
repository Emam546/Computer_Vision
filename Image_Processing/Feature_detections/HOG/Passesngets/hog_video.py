# Description: Detect pedestrians in a video using the
#   Histogram of Oriented Gradients (HOG) method

import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
import os

# Make sure the video file is in the same directory as your code
filename = '../../../../videos/Pedestrian_Walking.mp4'

cv2.namedWindow("Frame")


def main():
    os.chdir(os.path.dirname(__file__))
    # Create a HOGDescriptor object
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(filename)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def detect(val=None):
        pass

    cv2.createTrackbar("TRACK_BAR", "Frame", 100, 500, detect)
    for idx in range(0, frames_num, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break

        detect()
        val = cv2.getTrackbarPos("TRACK_BAR", "Frame")
        (bounding_boxes, weights) = hog.detectMultiScale(frame, winStride=(10, 10), padding=(4, 4),
                                                         scale=val/100)
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
