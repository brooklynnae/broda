#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'road_images_1/img_{i}.jpg') for i in range(21)]

uh = 255
us = 20
uv = 255
lh = 0
ls = 0
lv = 220

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

for image in imgs:
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width/2), int(height/2)))
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    cv2.imshow('hsv thresholded image', np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray_img, 250, 255)
    cv2.imshow('white mask', np.hstack((image, cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))))

    cv2.waitKey(0)