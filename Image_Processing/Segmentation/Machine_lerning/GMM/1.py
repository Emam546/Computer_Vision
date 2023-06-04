import numpy as np
import cv2
import os
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(__file__))
img = cv2.imread("./BSE.tif")
cv2.imshow("ORG_IMAGE", img)

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  # -1 reshape means, in this case MxN

gmm_model = GMM(n_components=4, covariance_type='tied').fit(img2)
gmm_labels = gmm_model.predict(img2)
# Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
cv2.imshow("RESULT", segmented)

n = 4
gmm_model = GMM(n, covariance_type='tied').fit(img2)
bic_value = gmm_model.bic(img2)
print(bic_value)
cv2.waitKey(0)
cv2.destroyAllWindows()
n_components = np.arange(1, 10)
gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]
plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')
plt.show()