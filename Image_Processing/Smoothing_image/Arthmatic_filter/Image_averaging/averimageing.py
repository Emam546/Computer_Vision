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

def video_median(cap, sample_num=25):
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_num = min(frames_num, sample_num)
    # Store selected frames in an array
    sample_image = []

    for fid in range(0, frames_num, int(frames_num/sample_num)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            sample_image.append(frame)
    return np.median(sample_image, axis=0).astype(dtype=np.uint8)

def video_mean(cap, sample_num=25):
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_num = min(frames_num, sample_num)

    # frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    ret, frame = cap.read()
    # Calculate the median along the time axis
    if ret:
        sample_image = np.zeros_like(frame, np.float16)
        # Store selected frames in an array
        num_img = 0
        for fid in range(0, frames_num, int(frames_num/sample_num)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                sample_image += frame.astype(np.float16)
                num_img += 1

        # Calculate the median along the time axis
        # medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return (sample_image/num_img).astype("uint8")

def video_segmenter(cap, bg_img, diff=100, mode=0):
    if mode == 1:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    elif mode == 0:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    for wimg in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, wimg)
        ret, frame = cap.read()
        if ret:
            result_segment = np.zeros(bg_img.shape[:2], "uint8")
            if mode == 1:
                sub = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
                subtract_img = cv2.absdiff(bg_img, sub)
                result_segment[subtract_img > diff] = 255
            elif mode == 0:
                sub = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subtract_img = cv2.absdiff(bg_img, sub)
                result_segment[subtract_img > diff] = 255
            else:
                subtract_img = cv2.absdiff(bg_img, frame)
                for c in range(bg_img.shape[2]):
                    result_segment[subtract_img[:, :, c] > diff] = 255
            yield True, result_segment, frame
        else:
            return False, None, None
    raise StopIteration

def video_background(cap, sample_num=25):
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_num = min(frames_num, sample_num)
    # sample_num=math.ceil(math.sqrt(sample_num))**2
    # frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    ret, frame = cap.read()
    # Calculate the median along the time axis
    if ret:
        sample_image = np.zeros((*frame.shape[:2], 1, 3), "uint8")
        # hist=np.zeros((*frame.shape[:2],256,256,256),np.float16)
        # Store selected frames in an array

        for fid in range(0, frames_num, int(frames_num/sample_num)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                frame = frame.reshape((*frame.shape[:2], 1, 3))
                sample_image = np.concatenate([frame, sample_image], axis=2)

        # sample_image=cv2.normalize(sample_image,)
        result = np.zeros((*frame.shape[:2], 3), "uint8")
        # squrt=int(math.sqrt(sample_image[0,0].shape[0]) )

        for y in progressBar(range(frame.shape[0]), length=50):
            for x in range(frame.shape[1]):
                s_values, s_counts = np.unique(
                    sample_image[y, x], return_counts=True, axis=0)
                result[y, x] = s_values[np.argmax(s_counts)]

        return result

def __test_segmenter(cap, sample_num=25, fps=None, diff=100, mode=3):
    bg_img = video_median(cap, sample_num)
    if bg_img is None:
        return
    fps = cap.get(cv2.CAP_PROP_FPS) if fps == None else fps
    cv2.imshow("BACK_GROUND", bg_img)
    for ret, segmented_img, frame in video_segmenter(cap, bg_img, diff, mode):
        if ret:
            cv2.imshow('Normal image', frame)
            cv2.imshow('segmented_image', segmented_img)
            if cv2.waitKey(fps) == 0x1B:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    SAMPLES = [
        "./cars_video.mp4",
        "./slow_traffic_small.mp4",
        "./face_detector.mp4"]
    for video in SAMPLES:
        # __test_segmenter(cv2.VideoCapture(video),25,fps=1,diff=40,mode=0)
        cv2.imshow("Back_ground", video_background(
            cv2.VideoCapture(video), 25))
        cv2.waitKey(0)
