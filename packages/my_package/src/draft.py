#!/usr/bin/env python

import numpy as np
import cv2
import sys
import rospy
import math
import time
import os

#duckietown
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import Twist2DStamped
from image_geometry import PinholeCameraModel


class BackTrackingNode(object):

    def __init__(self):
        self.node_name = rospy.get_name()
        self.bridge = CvBridge()
        self.active = True
        self.currentImg = None
        
        self.publish_duration = rospy.Duration.from_sec(1.0/2.0)
        self.last_stamp = rospy.Time.now()
        
        #Subscribers and publishers
        self.imagesub = rospy.Subscriber("~image",CompressedImage, self.find_marker, buff_size=921600,queue_size=1)
        self.movepub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=10)
        self.imagepub = rospy.Publisher("~image", Image, queue_size=1)
        self.infosub = rospy.Subscriber("~camera_info", CameraInfo, self.get_camera_info, queue_size=1)
        self.camerainfo = PinholeCameraModel()
        
        rospy.loginfo("[%s] Initialization completed" % (self.node_name))
        
#get camera info for pinhole camera model
    def get_camera_info(self, camera_msg):
        self.camerainfo.fromCameraInfo(camera_msg)

#step 1 : find the back circle grids using cv2.findCirclesGrid
##### set (x,y) for points and flag for detection
    def find_marker(self, image_msg):
    
        if not self.active:
            return
        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        else:
            self.last_stamp = now
            
            
        try:
            self.currentImg = self.bridge.compressed_imgmsg_to_cv2(
                image_msg, "bgr8")
        except CvBridgeError as e:
            print e
        
        gray = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
        
        detection, corners = cv2.findCirclesGrid(gray,(7,3))
        
        copy = currentImg.copy()
        
        if detection:
            cv2.drawChessboardCorners(copy, (7,3), corners, detection)
            
            twoone = []
            for i in range(0, 21):
                point = [corners[i][0][0], corners[i][0][1]]
                twoone.append(point)
            twoone = np.array(twoone)
            
            self.originalmatrix()
            detection, rvec, tvec = self.gradient(twoone)
            distance(copy, tvec)
            
        cv2.imshow('detection', copy)
        cv2.wait(1)
        
#alternative step : find the back circle grids using double thresholds blob detector
##### set (x,y) for points and flag for detection
#    def find_marker2(self, image_msg):
    
 #       if not self.active:
#            return

#step 2 : makes matrix for 3d original shape
    def originalmatrix(self):
    #coners and points
        self.originalmtx = np.zeros([21, 3])
        for i in range(0, 7):
            for j in range(0, 3):
                self.originalmtx[i + j * 7, 0] = 0.0125 * i - 0.0125 * 3
                self.originalmtx[i + j * 7, 1] = 0.0125 * j - 0.0125
        

#step 3 : use intrinsic matrix and solvePnP, return rvec and tvec, print (show) norm
    def gradient(self, imgpts):
    #using solvePnP to find rotation vector and translation vector
        works, rvec, tvec = cv2.solvePnP(self.originalmtx, imgpts, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
        
        return works, rvec, tvec
        
        

#step 4 : find distance between robot and following robot print out distance and time
    #def distance(img,datas):
    #use tvec to calculate distance
        

"""
after step 4, wanted outcomes are

list shape of :

list[timeN, 3] ~ list[timeN, (distance, rvec, tvec)]


"""

#step 5 : use joy mapper to controll the robot
    #def move(self):
        #if detected = LF false, if not = LF true
        
        #if distance = 0, wait
        
        #if distance > 0 and length.listmovement > 5
        


def main(args):

    rospy.init_node('back_tracking', anonymous=True)

    back_tracking_node = BackTrackingNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

