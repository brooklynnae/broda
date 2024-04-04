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

        self.img = None
        self.min_sign_area = 6000
        
        self.sign_img = None

        self.firstSignTime = None
        self.durationBetweenSigns = rospy.Duration.from_sec(5)

    # callback function for robot camera feed 
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
    def check_if_sign(self, img):
        """
        Checks if a sign is present in the provided image.

        Args:
            img (numpy.ndarray): The image in which to check for a sign.

        Returns:
            numpy.ndarray or None: The cropped image of the sign if found, None otherwise.
        """

        # threshold camera image for blue
        uh_blue = 150; us_blue = 255; uv_blue = 255
        lh_blue = 5; ls_blue = 19; lv_blue = 0
        lower_hsv = np.array([lh_blue,ls_blue,lv_blue])
        upper_hsv = np.array([uh_blue,us_blue,uv_blue])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        # threshold camera image for white
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_mask1 = cv2.inRange(gray_img, 98, 105)
        white_mask2 = cv2.inRange(gray_img, 197, 205)
        white_mask3 = cv2.inRange(gray_img, 119, 125)
        white_mask = cv2.bitwise_or(white_mask1, white_mask2)
        white_mask = cv2.bitwise_or(white_mask, white_mask3)

        # combine masks
        blue_mask_not = cv2.bitwise_not(blue_mask)
        combined_mask = cv2.bitwise_and(blue_mask_not, white_mask)

        # find largest contour in the combined mask image
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0: # return None if no contours are found
            # print('no sign detected - no contours')
            return None
        largest_contour = max(contours, key=cv2.contourArea)

        # filter out contours that are too small
        area = cv2.contourArea(largest_contour)
        if area < self.min_sign_area:
            # print('no sign detected - too small')
            return None

        # find the corners of the sign
        x, y, w, h = cv2.boundingRect(largest_contour)
        overlapping_points = [(point[0][0], point[0][1]) for point in largest_contour if point[0][0] <= x+10 or point[0][0] >= x+w-11]
        corner1 = min((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
        corner2 = min((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())
        corner3 = max((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
        corner4 = max((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

        # perspective transform and crop the image
        src_pts = np.array([corner1, corner2, corner3, corner4], dtype='float32')
        dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped_img = cv2.warpPerspective(img, matrix, (w, h))

        # threshold for red in the cropped image
        uh_red = 130; us_red = 255; uv_red = 255
        lh_red = 120; ls_red = 100; lv_red = 50
        lower_hsv_crop = np.array([lh_red, ls_red, lv_red])
        upper_hsv_crop = np.array([uh_red, us_red, uv_red])
        hsv_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        red_mask_cropped = cv2.inRange(hsv_cropped_img, lower_hsv_crop, upper_hsv_crop)

        # filter out if no red in the cropped image
        if not np.any(red_mask_cropped):
            # print('no sign detected - no red')
            return None
        
        # TODO: add code to see if sign is cut off 

        # found a sign !!
        # print('sign detected')
        return cropped_img
    
    def compare_sign(self, new_sign):
        """
        Compares the new sign image to the currently stored sign image. If the new image is 
        larger, it replaces the stored image.

        Args:
            new_sign (numpy.ndarray): The new sign image to compare with the currently stored sign image.

        Returns:
            None
        """
        if self.sign_img is None: # if stored sign image hasn't been assigned yet, assign it
            self.sign_img = new_sign
            self.firstSignTime = rospy.Time.now() # start timer for reading sign
            print('assigned sign image and timer started')
        else:
            if new_sign.size > self.sign_img.size: # compare size of new sign to stored sign
                self.sign_img = new_sign
                print('bigger sign found')
        return

    # when enough time has elapsed from initial sign detection, get the letters from the best 
    # sign image and send them to the neural network
    def read_sign(self):
        # TODO: crop sign to letters and send to NN
        print('sent sign image to NN')
        return

    def run(self):
        while not rospy.is_shutdown():
            # check if robot camera feed sees a sign
            if self.img is not None:
                cropped_img = self.check_if_sign(self.img) # returns None if no sign detected
                if cropped_img is not None:
                    self.compare_sign(cropped_img) # changes self.sign_img if new sign is larger

            # if we've found a sign ...
            if self.sign_img is not None:
                # display the sign image
                cv2.imshow('sign', self.sign_img)
                cv2.waitKey(1)
                # check if enough time has elapsed to read the sign
                current_time = rospy.Time.now()
                elapsed_time = current_time - self.firstSignTime
                if elapsed_time > self.durationBetweenSigns:
                    self.read_sign()
                    self.sign_img = None
                    self.firstSignTime = None

            rospy.sleep(0.1) # 100ms delay

if __name__ == '__main__':
    try:
        my_bot = SignReader()
        rospy.sleep(1)
        my_bot.run()
    except rospy.ROSInterruptException:
        pass
