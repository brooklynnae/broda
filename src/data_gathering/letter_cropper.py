#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'img_{i}.jpg') for i in range(8)]
# shrink images
width = imgs[0].shape[1] 
height = imgs[0].shape[0] 
imgs = [cv2.resize(img, (int(width/3), int(height/3))) for img in imgs]

# put images all into one
ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
img_array = np.vstack((ho1, ho2))
# cv2.imshow('img1', img_array)

# # v1
# uh = 140
# us = 255
# uv = 205
# lh = 85
# ls = 65
# lv = 0

# v2
uh = 114
us = 255
uv = 227
lh = 5
ls = 19
lv = 0

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

# hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
# cv2.imshow('thresholded image', mask)

# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_array, contours, -1, (0, 255, 0), cv2.FILLED)
# cv2.imshow('contours', img_array)

# cv2.waitKey(0)

for i in range(8):
    img = cv2.imread(f'img_{i}.jpg')
    # cv2.imshow('image', img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    # cv2.imshow('thresholded image', mask)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign_mask1 = cv2.inRange(gray_img, 95, 105)
    sign_mask2 = cv2.inRange(gray_img, 195, 205)
    sign_mask3 = cv2.inRange(gray_img, 115, 125)
    sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
    sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)
    # cv2.imshow('sign_mask', sign_mask)

    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours', img)

    # countours, _ = cv2.findContours(sign_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, countours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours', img)

    mask_not = cv2.bitwise_not(mask)
    combined_mask = cv2.bitwise_and(mask_not, sign_mask)
    cv2.imshow('combined_mask', combined_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    cv2.imshow('cropped_img1', cropped_img)
    src_pts = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img2 = cv2.warpPerspective(img, M, (w, h))
    cv2.imshow('cropped_img_perspective', cropped_img2)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

# cv2.imwrite('all_imgs.jpg', img_array) 