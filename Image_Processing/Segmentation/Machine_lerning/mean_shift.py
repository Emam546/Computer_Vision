import cv2
import os
import numpy as np
from sklearn.cluster import MeanShift
os.chdir(os.path.dirname(__file__))
img = cv2.imread("../../../images/messi5.jpg", 0)
Z = img.reshape(1, -1)
# convert to np.float32
# Z = np.float32(img)
# print(np.unique(Z,axis=1).shape)
mean_shift = MeanShift()
mean_shift.fit(Z)
center = mean_shift.cluster_centers_.astype(np.int16)
print(mean_shift.cluster_centers_.shape)
print(Z.shape)
print(mean_shift.labels_)

res = center[mean_shift.labels_.flatten()]
res2 = res.reshape((img.shape)).astype(np.uint8)
cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
