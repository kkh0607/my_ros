#!/usr/bin/env python

import os
import numpy as np
import cv2
import sys
import rospy
import math
import time
import rosbag
from duckietown import DTROS
from std_msgs.msg import String, Int32
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from duckietown_msgs.msg import Twist2DStamped

class LeaderDrive(DTROS):
    def __init__(self, node_name):
    # initialize the DTROS parent class
        super(LeaderDrive, self).__init__(node_name=node_name)
        self.count = 0
        self.max = 100
        self.on = 0
        self.pub_move = rospy.Publisher("/duckiesam/joy_mapper_node/car_cmd", Twist2DStamped, queue_size = 1)
        
    def move(self):
        #publish movement
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        
    
    def start(self):
        self.sub_move = rospy.Subscriber("/duckiesam/joy_mapper_node/car_cmd", Twist2DStamped, self.bagout queue_size = 1)
    
    def bagout(self):
        #publish bag
        bag = rosbag.Bag('bag'+self.count + '.bag','w')
        
    

if __name__ == '__main__':
    # create the node
    node = LeaderDrive(node_name='leader_drive')

    # keep spinning
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")

    cv2.destroyAllWindows()
