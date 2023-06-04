import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def compute_gradients(firstImage, secondImage):
    """
    Compute gradients in x, y and t direction between images
    :param firstImage: First image
    :param secondImage: Second image
    :return: Gradients
    """

    firstImage = firstImage / 255
    secondImage = secondImage / 255

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    # kernel_x = np.array([[-1., 1.], [-1., 1.]]) / 4
    # kernel_y = np.array([[-1., -1.], [1., 1.]]) / 4
    # kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

    Ix = scipy.ndimage.convolve(input=firstImage, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=firstImage, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode="nearest"
    )

    I = [Ix, Iy, It]

    return I


def horn_schunk(firstImage, secondImage, lambada, num_iter, image_ind, dataset):
    """
    Horn Schunk Optical flow estimation between firstImage and secondImage
    :param firstImage: First image
    :param secondImage: Second Image
    :param lambada: Regularization parameter
    :param num_iter: Number of iterations of optimization
    :param image_ind: Current image index
    :param dataset: Dataset name
    :return: Optical flow, Gradients
    """

    u = np.zeros([firstImage.shape[0], firstImage.shape[1]])
    v = np.zeros([firstImage.shape[0], firstImage.shape[1]])

    [Ix, Iy, It] = compute_gradients(firstImage, secondImage)

    # Optical flow averaging kernel
    kernel = np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]], dtype=np.float32)

    for _ in range(num_iter):
        u_avg = scipy.ndimage.convolve(input=u, weights=kernel, mode="nearest")
        v_avg = scipy.ndimage.convolve(input=v, weights=kernel, mode="nearest")

        grad = (Ix * u_avg + Iy * v_avg + It) / (lambada ** 2 + Ix ** 2 + Iy ** 2)

        u = u_avg - lambada * Ix * grad
        v = v_avg - lambada * Iy * grad

    flow_map = compute_flow_map(u, v, 8)
    plt.imshow(firstImage * 255, cmap="gray")
    plt.imshow(flow_map, cmap=None)

    # Plot, visualize and save the optical flow
    # added_image = cv2.addWeighted(firstImage, 0.5, flow_map, 1, 0, dtype=cv2.CV_32F)
    # cv2.imwrite(f'./results/horn_schunck/optical_flow/{dataset}/flow_map_{image_ind}.png', added_image)
    plt.show()

    flow = [u, v]
    I = [Ix, Iy, It]

    return flow, I