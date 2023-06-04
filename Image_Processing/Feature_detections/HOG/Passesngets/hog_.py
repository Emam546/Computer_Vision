import cv2
import os
# Import the OpenCV library to enable computer vision

# Description: Detect pedestrians in an image using the
#   Histogram of Oriented Gradients (HOG) method

# Make sure the image file is in the same directory as your code
filename = './pedestrians_2.jpg'


def main():
    os.chdir(os.path.dirname(__file__))
    # Create a HOGDescriptor object
    hog = cv2.HOGDescriptor()

    # Initialize the People Detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load an image
    image = cv2.imread(filename)

    # Detect people
    # image: Source image
    # winStride: step size in x and y direction of the sliding window
    # padding: no. of pixels in x and y direction for padding of sliding window
    # scale: Detection window size increase coefficient
    # bounding_boxes: Location of detected people
    # weights: Weight scores of detected people
    (bounding_boxes, weights) = hog.detectMultiScale(image,
                                                     winStride=(4, 4),
                                                     padding=(8, 8),
                                                     scale=1.05)

    # Draw bounding boxes on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image,
                      (x, y),
                      (x + w, y + h),
                      (0, 0, 255),
                      4)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
