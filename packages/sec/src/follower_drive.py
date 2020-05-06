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

class FollowerDrive(DTROS):
    def __init__(self, node_name):
    # initialize the DTROS parent class
    super(FollowerDrive, self).__init__(node_name=node_name)

if __name__ == '__main__':
    # create the node
    node = FollowerDrive(node_name='follower_drive')
    
    # keep spinning
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")

    cv2.destroyAllWindows()
