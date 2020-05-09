#!/usr/bin/env python

import os
import numpy as np
import cv2
import sys
import rospy
import math
import time
from mutex import mutex
from duckietown import DTROS
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from duckietown_msgs.msg import Twist2DStamped, BoolStamped


class MyNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name)

    # construct publisher and subsriber
        self.pub = rospy.Publisher('/duckiesam/chatter', String, queue_size=10)
        self.sub_image = rospy.Subscriber("/duckiesam/camera_node/image/compressed", CompressedImage, self.find_marker, buff_size=921600,queue_size=1)
        self.pub_image = rospy.Publisher('/duckiesam/modified_image', Image, queue_size = 1)
        self.sub_info = rospy.Subscriber("/duckiesam/camera_node/camera_info", CameraInfo, self.get_camera_info, queue_size=1)
        self.pub_move = rospy.Publisher("/duckiesam/joy_mapper_node/car_cmd", Twist2DStamped, queue_size = 1)
        self.leader_detected = rospy.Publisher("/duckiesam/detection",BoolStamped, queue_size=1)

	#values for detecting marker
        self.starting = 0
        self.ending = 0
        self.camerainfo = PinholeCameraModel()
        self.bridge = CvBridge()
        self.gotimage = False
        self.imagelast = None
        self.processedImg = None
        self.detected = False

	#values for calculating pose of robot
	self.originalmatrix()
        self.solP = False
        #self.rotationvector = None
        #self.translationvector = None
        self.axis = np.float32([[0.0125,0,0], [0,0.0125,0], [0,0,-0.0375]]).reshape(-1,3)
        #self.distance = None
        #self.angle_f = None
        #self.angle_l = None

	#values for driving the robot
        self.maxdistance = 0.2
        self.speedN = 0
        self.e_vB = 0
        self.rotationN = 0
        self.mindistance = 0.1
        self.d_e = 0 #distance error
        self.d_e_1 = 5
        self.d_e_2 = 10
        self.y2 = 0
        self.controltime = rospy.Time.now()
        self.Kp = 1
        self.Ki = 0.1
        self.Kd = 0
        self.I = 0
        self.Rp = 1
        self.Ri = 1
        self.Rd = 1

	rospy.on_shutdown(self.my_shutdown)
        
    def initialvalues(self):
        self.default_v = 0.20
        self.maxdistance = 0.2
        self.speedN = 0
        self.e_vB = 0
        self.rotationN = 0
        self.mindistance = 0.2
        self.d_e = 0 #distance error
        self.d_e_1 = 5
        self.d_e_2 = 10
        self.y2 = 0
        self.controltime = rospy.Time.now()
        self.Kp = 1
        self.Ki = 0.1
        self.Kd = 0
        self.I = 0
        self.Rp = 1
        self.Ri = 1
        self.Rd = 1
        
        
    #get camera info for pinhole camera model
    def get_camera_info(self, camera_msg):
        self.camerainfo.fromCameraInfo(camera_msg)

    #step 1 : find the back circle grids using cv2.findCirclesGrid
    ##### set (x,y) for points and flag for detection
    def find_marker(self, image_msg):
        try:
            self.imagelast = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        if self.gotimage == False:
            self.gotimage = True
         
        #time checking
        self.starting = rospy.Time.now()
        #from_last_image = (self.starting - self.ending).to_sec()
        
        gray = cv2.cvtColor(self.imagelast, cv2.COLOR_BGR2GRAY)
        
        detection, corners = cv2.findCirclesGrid(gray,(7,3))
        
        processedImg = self.imagelast.copy()
        cmd = Twist2DStamped()
        cmd.header.stamp = self.starting

        if detection:
            cv2.drawChessboardCorners(processedImg, (7,3), corners, detection)
            self.detected = True
            #self.controltime = rospy.Time.now()
            twoone = []
            for i in range(0, 21):
                point = [corners[i][0][0], corners[i][0][1]]
                twoone.append(point)
            twoone = np.array(twoone)
            
            
            rotationvector, translationvector, processedImg = self.gradient(twoone,processedImg)
            self.detected = self.solP
            img_out = self.bridge.cv2_to_imgmsg(processedImg, "bgr8")
            self.pub_image.publish(img_out)
            distance, angle_f, angle_l, y2 = self.find_distance(rotationvector, translationvector)
            self.move(y2, angle_l, distance)
            self.ending = rospy.Time.now()
        else:
            self.detected = False
            img_out = self.bridge.cv2_to_imgmsg(self.imagelast, "bgr8")
            self.pub_image.publish(img_out)
            self.ending = rospy.Time.now()
            cmd.v = 0
            cmd.omega = 0
            self.pub_move.publish(cmd)
            
    #step 2 : makes matrix for 3d original shape
    def originalmatrix(self):
    #coners and points
        self.originalmtx = np.zeros([21, 3])
        for i in range(0, 7):
            for j in range(0, 3):
                self.originalmtx[i + j * 7, 0] = 0.0125 * i - 0.0125 * 3
                self.originalmtx[i + j * 7, 1] = 0.0125 * j - 0.0125

    
    #step 3 : use intrinsic matrix and solvePnP, return rvec and tvec, print axis
    def gradient(self, imgpts, processedImg):
    #using solvePnP to find rotation vector and translation vector and also find 3D point to the image plane
        solP, rotationvector, translationvector = cv2.solvePnP(self.originalmtx, imgpts, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
        if solP:
            pointsin3D, jacoB = cv2.projectPoints(self.originalmtx, rotationvector, translationvector, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
            pointaxis, _ = cv2.projectPoints(self.axis, rotationvector, translationvector, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[0].ravel()), (255, 0, 0), 2)
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[1].ravel()), (0, 255, 0), 2)
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[2].ravel()), (0, 0, 255), 3)
	    #textdistance = "x = %s, y = %s, z = %s" % (self.distance, self.angle_f, self.angle_l, self.y2)
            #rospy.loginfo("%s" % textdistance)
	return rotationvector, translationvector, processedImg

    #step 4 : find distance between robot and following robot print out distance and time
    def find_distance(self, translationvector, rotationvector):
    #use tvec to calculate distance
        tvx = translationvector[0]
        tvy = translationvector[1]
        tvz = translationvector[2]
        
        distance = math.sqrt(tvx*tvx + tvz*tvz)
        angle_f = np.arctan2(tvx[0],tvz[0])

        R, _ = cv2.Rodrigues(rotationvector)
        R_inverse = np.transpose(R)
        angle_l = np.arctan2(-R_inverse[2,0], math.sqrt(R_inverse[2,1]**2 + R_inverse[2,2]**2))
        
        T = np.array([-np.sin(angle_l), np.cos(angle_l)])
        tvecW = -np.dot(R_inverse, translationvector)
        x_y = np.array([tvz[0], tvx[0]])
        
        y2 = -np.dot(T,x_y) - 0.01*np.sin(angle_l)
        
        textdistance = "Distance = %s, Angle of Follower = %s, Angle of Leader = %s, y = %s" % (distance, angle_f, angle_l, y2)
        rospy.loginfo("%s" % textdistance)
        #self.pub.publish(textdistance)
	return distance, angle_f, angle_l, y2
        
    #step 5 : use joy mapper to control the robot PID controller
    def move(self, y_to, angle_to, d):
        #y_to is needed y value to be parallel to leader's center line
        #angle_to is angle needed to rotate
        #d is distance between required position and now position
        cmd = Twist2DStamped()
        
        time = rospy.Time.now()
        cmd.header.stamp = time
        dt = (time - self.controltime).to_sec()
        if dt > 3:
            if d < self.maxdistance:
                cmd.v = 0
                cmd.omega = 0
        else:
            self.d_e = d - self.maxdistance
            error_d = (self.d_e - self.d_e_1)/dt
            errorB = (self.d_e_1 - self.d_e_2)/dt
            
            e_v = error_d - errorB
            #PID controller for the angle, velocity is default or 0.
            
            P = self.Kp*(e_v)
            self.I = self.I + self.Ki*(e_v + self.e_vB)/2*dt
            D = self.Kd*(e_v + self.e_vB)/dt
            
            #self.speedN = P + self.I + D
            
            self.rotationN = self.Rp*(y_to) + self.Ri*(angle_to) + self.Rd*(np.sin(angle_to))
            
            cmd.v = self.default_v
            cmd.omega = self.rotationN

            self.e_vB = e_v
            self.d_e_2 = self.d_e_1
            self.d_e_1 = self.d_e
            if self.d_e < 0.05 or self.speedN < 0:
                cmd.v = 0
                cmd.omega = 0

        textdistance = "Velocity = %s, Rotation = %s" % (cmd.v, cmd.omega)
        rospy.loginfo("%s" % textdistance)
        self.pub_move.publish(cmd)
        self.controltime = time

    def my_shutdown(self):
	cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.pub_move.publish(cmd)
        rospy.sleep(1)
        print("shutting down") 


if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='my_node')
    
    # keep spinning
   # try:
    rospy.spin()
    #except KeyboardInterrupt:
    #    print("shutting down")
	
    #cv2.destroyAllWindows()
    
