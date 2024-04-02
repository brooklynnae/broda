#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class SignReader():
    def __init__(self):
        rospy.init_node('sign_reader')

        self.bridge = CvBridge()
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        uh = 150
        us = 255
        uv = 255
        lh = 5
        ls = 19
        lv = 0
        self.lower_hsv = np.array([lh,ls,lv])
        self.upper_hsv = np.array([uh,us,uv])
        self.img = None

        self.foundSign = False
                
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.crop(self.img)

    def crop(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, self.lower_hsv, self.upper_hsv)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sign_mask1 = cv2.inRange(gray_img, 98, 105)
        sign_mask2 = cv2.inRange(gray_img, 197, 205)
        sign_mask3 = cv2.inRange(gray_img, 119, 125)
        sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
        sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)

        mask_not = cv2.bitwise_not(mask)
        combined_mask = cv2.bitwise_and(mask_not, sign_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if not(100 < w < 260) or not(70 < h < 170):
            # print('no sign detected')
            if self.foundSign:
                cv2.destroyWindow('cropped_img')
                self.foundSign = False
            return
        else:
            print("sign detected")
            overlapping_points = [(point[0][0], point[0][1]) for point in largest_contour if point[0][0] <= x+10 or point[0][0] >= x+w-11]
            corner1 = min((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
            corner2 = min((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())
            corner3 = max((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
            corner4 = max((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

            src_pts = np.array([corner1, corner2, corner3, corner4], dtype='float32')
            dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            cropped_img = cv2.warpPerspective(img, matrix, (w, h))
            
            # final_img = cv2.resize(cropped_img, (w*4, h*4))
            # cv2.imshow('final_img', final_img)

            self.foundSign = True

            cv2.imshow('cropped_img', cropped_img)
            cv2.waitKey(1)
            return

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        my_bot = SignReader()
        rospy.sleep(1)
        my_bot.start()
    except rospy.ROSInterruptException:
        pass

