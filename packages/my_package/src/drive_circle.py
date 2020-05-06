#!/usr/bin/env python

import os
import numpy as np
import cv2
import sys
import rospy
import math
import time
from duckietown import DTROS
from duckietown_msgs.msg import Twist2DStamped, BoolStamped

class DriveCircle(DTROS):
    def __init__(self, node_name):
    # initialize the DTROS parent class
        super(DriveCircle, self).__init__(node_name=node_name)
    
        self.pub_move = rospy.Publisher('~car_cmd', Twist2DStamped, queue_size = 1)
        #self.ready = rospy.Subscriber('/duckie99/my_node/detection', self.drive, queue_size=1)
        self.startingtime = rospy.Time.now()
        self.defaultvelocity = 0.22
        self.defaultomega = 0.6
	#init
	car_control_msg = Twist2DStamped()
        car_control_msg.v = 0
        car_control_msg.omega = 0
        self.pub_move.publish(car_control_msg)
	rospy.sleep(1)
        

    def drive_line(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = self.defaultvelocity
        car_control_msg.omega = self.defaultomega
        self.pub_move.publish(car_control_msg)
        
        #car_control_msg.v = 0.0
        #self.pub_move.publish(car_control_msg)
     
    def turn_right(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 1.0
        self.pub_move.publish(car_control_msg)
        
    
    def drive_curve(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = self.defaultvelocity
        car_control_msg.omega = self.defaultomega
        self.pub_move.publish(car_control_msg)
        #rospy.sleep(1)
        
    def stop(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0
        car_control_msg.omega = 0
        self.pub_move.publish(car_control_msg)
	rospy.sleep(10)

    def drive(self):
	while not rospy.is_shutdown():
            self.drive_curve()
	    self.stop()
        

        
if __name__ == '__main__':
    # create the node
    node = DriveCircle(node_name='drive_circle')
    # run node
    node.drive()
    # keep spinning
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")

    cv2.destroyAllWindows()
