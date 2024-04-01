#! /usr/bin/env python3

import cv2
import numpy as np

# This code is kinda all over the place, it includes some stuff that was run to 
# put a whole bunch of images into one big image to test thresholding, as well 
# as the code to crop each image to the sign and perspective transform it.

# 1. Code for combining all images into one

imgs = [cv2.imread(f'sign_images_1/img_{i}.jpg') for i in range(8)]
imgs.append(cv2.imread('img_1.jpg'))
imgs.append(cv2.imread('img_6.jpg'))
imgs.append(cv2.imread('img_7.jpg'))
imgs.append(cv2.imread('img_12.jpg'))
imgs.append(cv2.imread('img_15.jpg'))
imgs.append(cv2.imread('img_18.jpg'))
imgs.append(cv2.imread('img_21.jpg'))
imgs.append(cv2.imread('img_22.jpg'))
# shrink images
width = imgs[0].shape[1] 
height = imgs[0].shape[0] 
imgs = [cv2.resize(img, (int(width/3), int(height/3))) for img in imgs]

# put images all into one
ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
ho3 = np.hstack((imgs[8], imgs[9], imgs[10], imgs[11]))
# ho4 = np.hstack((imgs[12], imgs[13], imgs[14], imgs[15]))
img_array = np.vstack((ho1, ho2, ho3))
# cv2.imshow('img1', img_array)
# cv2.waitKey(0)

# cv2.imwrite('img_array_1.jpg', img_array)
# cv2.imwrite('img_array_2.jpg', img_array)

# # v1
# uh = 140
# us = 255
# uv = 205
# lh = 85
# ls = 65
# lv = 0

# # v2
# uh = 114
# us = 255
# uv = 227
# lh = 5
# ls = 19
# lv = 0

# v3
uh = 150
us = 255
uv = 255
lh = 5
ls = 19
lv = 0

# # v4 test
# uh = 130
# us = 255
# uv = 255
# lh = 118
# ls = 0
# lv = 0

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
# cv2.imshow('thresholded image', mask)

# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_array, contours, -1, (0, 255, 0), cv2.FILLED)
# cv2.imshow('contours', img_array)
# cv2.imwrite('all_imgs.jpg', img_array) 

gray_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_img_array', gray_img_array)
sign_mask1 = cv2.inRange(gray_img_array, 99, 105)
sign_mask2 = cv2.inRange(gray_img_array, 197, 205)
sign_mask3 = cv2.inRange(gray_img_array, 119, 125)
sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)
# cv2.imshow('sign_mask', sign_mask)

uh2 = 142
us2 = 36
uv2 = 221
lh2 = 113
ls2 = 0
lv2 = 99

lower_hsv2 = np.array([lh2,ls2,lv2])
upper_hsv2 = np.array([uh2,us2,uv2])

mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv2)

mask_not = cv2.bitwise_not(mask)
mask_not2 = cv2.bitwise_not(mask2)
# combined_mask = cv2.bitwise_and(mask_not, mask_not2)
# combined_mask = cv2.bitwise_and(mask2, sign_mask)
combined_mask = cv2.bitwise_and(mask_not, sign_mask)
# cv2.imshow('combined_mask', combined_mask)

cv2.waitKey(0)

# 2. Code for cropping each image to the sign
for i in range(8):
    img = cv2.imread(f'sign_images_1/img_{i}.jpg')
    # img = cv2.imread(f'sign_images_2/img_{i}.jpg')
    # cv2.imshow('image', img)

    # Threshold the whole sign such that it is black (all blue and white parts)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv) 
    # cv2.imshow('thresholded image', mask)

    # Threshold just the white part of the sign such that it is white
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign_mask1 = cv2.inRange(gray_img, 98, 105)
    sign_mask2 = cv2.inRange(gray_img, 197, 205)
    sign_mask3 = cv2.inRange(gray_img, 119, 125)
    sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
    sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)
    # cv2.imshow('sign_mask', sign_mask)
    
    # # Draw the contours of both thresholdings
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours1', img)
    # countours, _ = cv2.findContours(sign_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, countours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours2', img)

    # Combine the two masks
    mask_not = cv2.bitwise_not(mask)
    combined_mask = cv2.bitwise_and(mask_not, sign_mask)
    # cv2.imshow('combined_mask', combined_mask)

    # Find the largest contour and crop the image
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    print(f'----------- image {i} -----------')
    print(w, h)

    # Roughly find the corners of the sign
    
    # # My code:
    # overlapping_points = []
    # for point in largest_contour:
    #     if point[0][0] <= x+10 or point[0][0] >= x+w-11:
    #         stored_point = (point[0][0], point[0][1])
    #         overlapping_points.append(stored_point)
            # print(stored_point)
    # corner1 = ()
    # corner2 = ()
    # corner3 = ()
    # corner4 = ()
    # ymin1 = 10000
    # ymax1 = 0
    # ymin2 = 10000
    # ymax2 = 0
    # for point in overlapping_points:
    #     if point[0] <= x+10:
    #         if point[1] <= ymin1:
    #             ymin1 = point[1]
    #             corner1 = point
    #         if point[1] >= ymax1:
    #             ymax1 = point[1]
    #             corner3 = point
    #     elif point[0] >= x+w-11:
    #         if point[1] <= ymin2:
    #             ymin2 = point[1]
    #             corner2 = point
    #         if point[1] >= ymax2:
    #             ymax2 = point[1]
    #             corner4 = point

    # # Copilot's more compact code:
    overlapping_points = [(point[0][0], point[0][1]) for point in largest_contour if point[0][0] <= x+10 or point[0][0] >= x+w-11]

    corner1 = min((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
    corner2 = min((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

    corner3 = max((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
    corner4 = max((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

    # print(corner1, corner2, corner3, corner4)
    # print('----------- next image -----------')

    cv2.circle(img, corner1, 5, (0, 0, 255), -1)
    cv2.circle(img, corner2, 5, (0, 0, 255), -1)
    cv2.circle(img, corner3, 5, (0, 0, 255), -1)
    cv2.circle(img, corner4, 5, (0, 0, 255), -1)
    cv2.imshow('circle img', img)

    if w < 75 or h < 65 or w > 175 or h > 120:
        print('no sign detected')
        cv2.waitKey(0)
        continue
    
    print('sign detected')
    # Crop the image and perspective transform it
    cropped_img = img[y:y+h, x:x+w]
    # cv2.imshow('cropped_img', cropped_img)
    src_pts = np.array([corner1, corner2, corner3, corner4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img2 = cv2.warpPerspective(img, M, (w, h))
    cv2.imshow('cropped_img_perspective', cropped_img2)
    # final_img = cv2.resize(cropped_img2, (w*4, h*4))
    # cv2.imshow('final_img', final_img)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite(f'sign_images_1/cropped_img_{i}.jpg', final_img)
    # cv2.imwrite(f'sign_images_2/cropped_img_{i}.jpg', final_img)


