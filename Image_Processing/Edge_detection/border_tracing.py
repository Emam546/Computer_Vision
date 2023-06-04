import cv2
import numpy as np
CONNECTIVITY = ((1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1))


def new_direction(dir, plus):
    if plus:
        return (dir+1) % 8
    else:
        if dir % 2 == 0:
            return (dir+7) % 8
        else:
            return (dir+6) % 8


def border_tracing(img):
    output_img = np.zeros_like(img)
    current_pixel = list(zip(*np.where(img > 0)))[0]
    current_dir = 7
    while True:
        current_dir = new_direction(current_dir, False)
        y, x = current_pixel
        dx, dy = CONNECTIVITY[current_dir]
        if img[y+dy, x+dx] > 0:
            if output_img[y+dy, x+dx] == 0:
                output_img[y+dy, x+dx] = img[y+dy, x+dx]
                current_pixel = y+dy, x+dx
                # print("NEW PIXEL",current_dir)
            else:
                print("SAME DIRECTION")
                # we have returned to same edge so we reach to the edge
                return output_img
        else:
            for _ in range(len(CONNECTIVITY)):
                current_dir = new_direction(current_dir, True)
                dx, dy = CONNECTIVITY[current_dir]
                if img[y+dy, x+dx] > 0:
                    if output_img[y+dy, x+dx] == 0:
                        output_img[y+dy, x+dx] = img[y+dy, x+dx]
                        current_pixel = y+dy, x+dx
                        # print("GET NEW DIRECTION",current_dir)
                        break
                    else:
                        print("SAME POINT IN THE CIRCLE")
                        return output_img


def _main():
    orgImg = np.zeros((600, 700), "uint8")
    cv2.rectangle(orgImg, (200, 350), (400, 500), 255, -1)
    cv2.imshow("FILTERED IMAGE", border_tracing(orgImg))
    cv2.imshow("ORG IMAGE", orgImg)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
