import cv2
import os
import numpy as np
from scipy.signal import convolve2d
LAPLACIAN_MASK = np.array([
    [1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]
])
kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25
kernelT = np.ones((2, 2)) * 0.25


def compute_flow_map(u, v, gran=8):
    flow_map = np.zeros(u.shape, "uint8")
    for y in range(0, flow_map.shape[0], gran):
        for x in range(0, flow_map.shape[1], gran):
            if y % gran == 0 and x % gran == 0:
                dx = 3 * int(u[y, x])
                dy = 3 * int(v[y, x])
                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)
    return flow_map


def compute_optical_flow(U, V):
    hsv = np.zeros((*U.shape, 3), "uint8")
    # Sets image saturation to maximum
    hsv[..., 1] = 255
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(U, V)
    # Sets image hue according to the optical flow direction
    hsv[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


depth = cv2.CV_32F
SUM_KERNEL = np.ones((3, 3))


def horn_chunk_optical_flow(prevImg, nextImg, alpha=4, max_iter=40):
    fx = cv2.filter2D(prevImg, depth, kernelX) + \
        cv2.filter2D(nextImg, depth, kernelX)
    fy = cv2.filter2D(prevImg, depth, kernelY) + \
        cv2.filter2D(nextImg, depth, kernelY)

    ft = (nextImg.astype(np.float32)-prevImg.astype(np.float32))/2

    U = np.zeros_like(fx)
    V = np.zeros_like(fx)

    D = (fy**2)+(fx**2)+alpha**2
    e = float("inf")
    for _ in range(max_iter):
        Uavg = cv2.filter2D(U, depth, LAPLACIAN_MASK)
        Vavg = cv2.filter2D(V, depth, LAPLACIAN_MASK)
        P = (fx*Uavg)+(fy*Vavg)+ft
        der = P/D
        U = Uavg-(fx*der)
        V = Vavg-(fy*der)

    return U, V


def _lucucs_kanda_opticlaflow(prevImg, nextImg):
    ofx = cv2.filter2D(prevImg, depth, kernelX) + \
        cv2.filter2D(nextImg, depth, kernelX)
    ofy = cv2.filter2D(prevImg, depth, kernelY) + \
        cv2.filter2D(nextImg, depth, kernelY)

    oft = (nextImg.astype(np.float32)-prevImg.astype(np.float32))/4

    fx = cv2.filter2D(ofx**2, depth, SUM_KERNEL)
    fy = cv2.filter2D(ofy**2, depth, SUM_KERNEL)
    ft = cv2.filter2D(oft**2, depth, SUM_KERNEL)
    A = np.array([fx, fy])
    AT = np.transpose(A, 4)
    return (np.dot(
        np.dot(
            (np.dot(AT, A))**-1), AT), ft)
    A = np.array([
        [fx**2, fx*fy],
        [fx*fy, fy**2]
    ])
    M = np.where(np.absolute((A[0, 0]*A[1, 1])-(A[0, 1]**2)) == 0)
    fxt = -cv2.filter2D(ofx*oft, depth, SUM_KERNEL)
    fyt = -cv2.filter2D(ofy*oft, depth, SUM_KERNEL)
    A = 1/A
    np.nan_to_num(A, False)
    U, V = A[(0, 0)]*fxt+A[(0, 1)]*fyt, A[(1, 0)]*fxt+A[(1, 1)]*fyt

    return U, V


def _lucucs_kanda_opticlaflow_2(previous_frame, current_frame, window_size=15):
    # Convert frames to grayscale
    previous_frame = np.expand_dims(previous_frame, axis=2)
    current_frame = np.expand_dims(current_frame, axis=2)

    previous_gray = np.mean(previous_frame, axis=2)
    current_gray = np.mean(current_frame, axis=2)

    # Compute gradients
    kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    fx = np.convolve(previous_gray.flatten(), kernel.flatten(),
                     mode='same').reshape(previous_gray.shape)
    fy = np.convolve(previous_gray.flatten(), kernel.flatten()[
                     ::-1], mode='same').reshape(previous_gray.shape)
    ft = current_gray - previous_gray

    # Compute optical flow
    half_window = window_size // 2
    u = np.zeros_like(previous_gray)
    v = np.zeros_like(previous_gray)

    for i in range(half_window, previous_gray.shape[0] - half_window):
        for j in range(half_window, previous_gray.shape[1] - half_window):
            Ix = fx[i - half_window:i + half_window + 1, j -
                    half_window:j + half_window + 1].flatten()
            Iy = fy[i - half_window:i + half_window + 1, j -
                    half_window:j + half_window + 1].flatten()
            It = ft[i - half_window:i + half_window + 1, j -
                    half_window:j + half_window + 1].flatten()

            A = np.vstack((Ix, Iy)).T
            b = -It

            # Solve the linear system of equations using pseudo-inverse
            if np.linalg.matrix_rank(A) >= 2:
                flow = np.linalg.pinv(A) @ b
                u[i, j] = flow[0]
                v[i, j] = flow[1]

    return u, v


def lucas_kanade_optical_flow(previous_frame, current_frame, window_size=15):
    # Reshape frames to include a color channel dimension
    previous_frame = np.expand_dims(previous_frame, axis=2)
    current_frame = np.expand_dims(current_frame, axis=2)

    # Convert frames to grayscale
    previous_gray = np.mean(previous_frame, axis=2)
    current_gray = np.mean(current_frame, axis=2)

    # Compute gradients
    kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    fx = convolve2d(previous_gray, kernel, mode='same')
    fy = convolve2d(previous_gray, kernel.T, mode='same')
    ft = current_gray - previous_gray

    # Compute optical flow
    half_window = window_size // 2
    u = np.zeros_like(previous_gray)
    v = np.zeros_like(previous_gray)

    # Create sliding windows
    Ix_windows = np.lib.stride_tricks.sliding_window_view(
        fx, (window_size, window_size))
    Iy_windows = np.lib.stride_tricks.sliding_window_view(
        fy, (window_size, window_size))
    It_windows = np.lib.stride_tricks.sliding_window_view(
        ft, (window_size, window_size))

    # Reshape windows for vectorized operations
    Ix_windows = Ix_windows.reshape(-1, window_size ** 2)
    Iy_windows = Iy_windows.reshape(-1, window_size ** 2)
    It_windows = It_windows.reshape(-1, window_size ** 2)

    # Solve the linear system of equations using pseudo-inverse
    A = np.column_stack((Ix_windows, Iy_windows))
    b = -It_windows

    valid_points = np.linalg.matrix_rank(A) >= 2
    valid_indices = np.nonzero(valid_points)[0]

    if valid_indices.size > 0:
        valid_indices = np.unravel_index(valid_indices, previous_gray.shape)
        flow_vectors = np.linalg.pinv(A[valid_points]) @ b[valid_points]
        u[valid_indices] = flow_vectors[:, 0]
        v[valid_indices] = flow_vectors[:, 1]

    return u, v


def __main(f):
    os.chdir(os.path.dirname(__file__))
    cap = cv2.VideoCapture("../../videos/cars_video.mp4")
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
        U, V = f(old_gray, frame_gray)

        for i, p in enumerate(corners):
            x, y = _hold(p)
            dx, dy = round(U[y, x]), round(V[y, x])

            x, y = x+dx, y+dy
            corners[i] = (x, y)
            try:
                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)
            except:
                print(x, y)

        cv2.imshow("FRAME", frame)
        flow_map=compute_flow_map(U,V)
        cv2.imshow("FRAME_2",flow_map)
        old_gray = frame_gray.copy()
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    __main(_lucucs_kanda_opticlaflow_2)
