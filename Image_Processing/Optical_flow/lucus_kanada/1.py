import cv2,os
import numpy as np
import scipy

def lucas_kanade(firstImage, secondImage, N=3, tau=1e-3):
    """
    Lucas Kanade Optical flow estimation between firstImage and secondImage
    :param firstImage: First image
    :param secondImage: Second Image
    :param N: Block size N x N
    :param image_ind: Current image index
    :param dataset: Dataset name
    :param tau: Threshold parameter
    :return: Optical flow, Gradients
    """

    firstImage = firstImage / 255
    secondImage = secondImage / 255
    image_shape = firstImage.shape
    half_window_size = N // 2

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    # kernel_x = np.array([[-1., 1.], [-1., 1.]]) / 4
    # kernel_y = np.array([[-1., -1.], [1., 1.]]) / 4
    # kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

    Ix = scipy.ndimage.convolve(
        input=firstImage, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(
        input=firstImage, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode="nearest"
    )

    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    # Find Lucas Kanade OF for a block N x N with least squares solution
    for row_ind in range(half_window_size, image_shape[0] - half_window_size):
        for col_ind in range(half_window_size, image_shape[1] - half_window_size):
            Ix_windowed = Ix[
                row_ind - half_window_size: row_ind + half_window_size + 1,
                col_ind - half_window_size: col_ind + half_window_size + 1,
            ].flatten()
            Iy_windowed = Iy[
                row_ind - half_window_size: row_ind + half_window_size + 1,
                col_ind - half_window_size: col_ind + half_window_size + 1,
            ].flatten()
            It_windowed = It[
                row_ind - half_window_size: row_ind + half_window_size + 1,
                col_ind - half_window_size: col_ind + half_window_size + 1,
            ].flatten()

            A = np.asarray([Ix_windowed, Iy_windowed]).reshape(-1, 2)
            b = np.asarray(It_windowed)

            A_transpose_A = np.transpose(A) @ A

            A_transpose_A_eig_vals, _ = np.linalg.eig(A_transpose_A)
            A_transpose_A_min_eig_val = np.min(A_transpose_A_eig_vals)

            # Noise thresholding
            if A_transpose_A_min_eig_val < tau:
                continue

            A_transpose_A_PINV = np.linalg.pinv(A_transpose_A)
            w = A_transpose_A_PINV @ np.transpose(A) @ b

            u[row_ind, col_ind], v[row_ind, col_ind] = w

    # Plot, visualize and save the optical flow
    flow_map = compute_flow_map(u, v, 8)
    cv2.imshow("FLOW MAP", flow_map)

    flow = [u, v]
    I = [Ix, Iy, It]

    return flow, I


def compute_flow_map(u, v, gran=8):
    """
    Plot optical flow map
    """

    flow_map = np.zeros(u.shape)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx = 2 * int(u[y, x])
                dy = 2 * int(v[y, x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)

    return flow_map


def __test():
    os.chdir(os.path.dirname(__file__))
    cap = cv2.VideoCapture("../../../videos/slow_traffic_small.mp4")
    # Create some random colors
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.GaussianBlur(cv2.cvtColor(
        old_frame, cv2.COLOR_BGR2GRAY), (15, 15), 0)
    corners = cv2.goodFeaturesToTrack(old_gray, mask=None,
                                      maxCorners=100,
                                      qualityLevel=0.3,
                                      minDistance=7,
                                      blockSize=7)

    corners = list(corners.reshape(-1, 2).astype(np.int16))

    def _hold(p):
        return max(0, min(p[0], old_gray.shape[1]-1)), max(0, min(p[1], old_gray.shape[0]-1))
    while (1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (U, V), _ = lucas_kanade(old_gray, frame_gray)

        for i, p in enumerate(corners):
            x, y = _hold(p)
            dx, dy = round(U[y, x]), round(V[y, x])

            x, y = x+dx, y+dy
            corners[i] = (x, y)

            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

        cv2.imshow("FRAME", frame)
        flow_map = compute_flow_map(U, V)
        cv2.imshow("FRAME_2", flow_map)
        old_gray = frame_gray.copy()
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    __test()
