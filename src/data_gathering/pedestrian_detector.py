#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'pedestrian_images_1/img_{i}.jpg') for i in range(8)]

height, width = imgs[0].shape[:2]
imgs = [cv2.resize(img, (int(width/2), int(height/2))) for img in imgs]

# ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
# ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
ho1 = np.hstack((imgs[0], imgs[1]))
ho2 = np.hstack((imgs[2], imgs[4]))
img_array = np.vstack((ho1, ho2))
cv2.imshow('pedestrian images', img_array)
cv2.waitKey(0)

cv2.imwrite('pedestrian_images_1/pedestrian_array_1.jpg', img_array)
