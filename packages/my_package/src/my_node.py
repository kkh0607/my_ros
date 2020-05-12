#!/usr/bin/env python

import os
import numpy as np
import cv2
import sys
import rospy
import math
import time
from duckietown import DTROS
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from duckietown_msgs.msg import Twist2DStamped, BoolStamped, VehiclePose


class MyNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name)

    # construct publisher and subsriber
        self.pub = rospy.Publisher('/duckiesam/chatter', String, queue_size=10)
        self.sub_image = rospy.Subscriber("/duckiesam/camera_node/image/compressed", CompressedImage, self.find_marker, buff_size=921600,queue_size=1)
        self.pub_image = rospy.Publisher('/duckiesam/camera_node/image', Image, queue_size = 1)
        self.sub_info = rospy.Subscriber("/duckiesam/camera_node/camera_info", CameraInfo, self.get_camera_info, queue_size=1)
        self.pub_move = rospy.Publisher("/duckiesam/joy_mapper_node/car_cmd", Twist2DStamped, queue_size = 1)
	self.pub_pose = rospy.Publisher("/duckiesam/pose", VehiclePose, queue_size=1)

	#values for detecting marker
        self.starting = 0
        self.ending = 0
        self.camerainfo = PinholeCameraModel()
        self.bridge = CvBridge()
        self.gotimage = False
        self.imagelast = None
        self.processedImg = None
        self.detected = False
	self.number = 0

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
        self.initialvalues()

	rospy.on_shutdown(self.my_shutdown)
        
    def initialvalues(self):
        
        self.maxdistance = 0.25
        self.speedN = 0
        self.e_vB = 0
        self.rotationN = 0
        self.mindistance = 0.2
	self.d_before = 0.0
        self.d_e = 0 #distance error
        #self.d_e_1 = 0
        #self.d_e_2 = 0
        self.y2 = 0
        self.controltime = rospy.Time.now()
        self.Kp = 0.5
        self.Ki = 0.1
        self.Kd = 0.1
        self.I = 0
        self.r1 = 1
        self.r2 = 2
        self.r3 = 1
        
        
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
	self.number += 1
	cmd.header.seq = self.number
	
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
            distance, angle_f, angle_l, y2 = self.find_distance(translationvector, rotationvector)
            self.move(y2, angle_l, distance, self.number)
            self.ending = rospy.Time.now()
        else:
            self.detected = False
            img_out = self.bridge.cv2_to_imgmsg(self.imagelast, "bgr8")
            self.pub_image.publish(img_out)
            self.ending = rospy.Time.now()
            cmd.v = 0
            cmd.omega = 0
            self.pub_move.publish(cmd)

	time_took = (self.ending - self.starting).to_sec()
	textdistance = "Num = %s, time = %s, Detected = %s" % (self.number, time_took, self.detected)
        rospy.loginfo("%s" % textdistance)
	rospy.Rate(10).sleep()
            
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
        self.solP, rotationvector, translationvector = cv2.solvePnP(self.originalmtx, imgpts, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
        if self.solP:
            pointsin3D, jacoB = cv2.projectPoints(self.originalmtx, rotationvector, translationvector, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
            pointaxis, _ = cv2.projectPoints(self.axis, rotationvector, translationvector, self.camerainfo.intrinsicMatrix(), self.camerainfo.distortionCoeffs())
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[0].ravel()), (255, 0, 0), 2)
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[1].ravel()), (0, 255, 0), 2)
            processedImg = cv2.line(processedImg, tuple(imgpts[10].ravel()), tuple(pointaxis[2].ravel()), (0, 0, 255), 3)
	    
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

	#tvecW is position of camera(follower vehicle) in world frame
        tvecW = -np.dot(R_inverse, translationvector)

	#desire point [0.20, 0] x,y, now tvz = x tvx = y
        x_y = np.array([tvecW[2][0], tvecW[0][0]])
        
        y2 = tvecW[0][0] - 0.05*np.sin(angle_l)
        
        textdistance = "Distance = %s, Angle of Follower = %s, Angle of Leader = %s, y = %s" % (distance, angle_f, angle_l, y2)
        rospy.loginfo("%s" % textdistance)
	position = VehiclePose()
	position.header.stamp = rospy.Time.now()
	position.rho.data = distance
	position.theta.data = angle_f
	position.psi.data = angle_l
	self.pub_pose.publish(position)
        #self.pub.publish(textdistance)
	return distance, angle_f, angle_l, y2
        
    #step 5 : use joy mapper to control the robot PID controller and steering angle by tracking error
    def move(self, y_to, angle_to, d, number):
        #y_to is needed y value to be parallel to leader's center line
        #angle_to is angle needed to rotate
        #d is distance between required position and now position
        cmd = Twist2DStamped()
        
        time = rospy.Time.now()
        cmd.header.stamp = time
	cmd.header.seq = number
        dt = (time - self.controltime).to_sec()
        if dt > 3 and not self.detected:
            #if d < self.maxdistance:
            cmd.v = 0
            cmd.omega = 0
	    self.initialvalues()
        else:
            self.d_e = d - self.mindistance #new distance error
            
	    error_d = (d - self.d_before)/dt + self.d_e/dt
            
            #e_v = error_d - errorB
            #PID controller for the velocity
            
            P = self.Kp*(error_d) #Proportional Controller 
            self.I = self.I + self.Ki*((error_d + self.e_vB)/2)*dt #Integral 
            D = self.Kd*(error_d + self.e_vB)/dt #
            
            self.speedN = P + self.I + D
            
	    #steering by longitudinal and tracking error angle with 
	    #r1,r2,r3 control parameters
            self.rotationN = self.r1*(y_to) + self.r2*(-angle_to) + self.r3*(np.sin(-angle_to))
            
            cmd.v = self.speedN
            cmd.omega = self.rotationN
	    textdistance = "Velocity = %s, Rotation = %s" % (cmd.v, cmd.omega)
            rospy.loginfo("%s" % textdistance)

            self.e_vB = error_d
            
	    self.d_before = d
            
            if d < 0.15 or self.speedN > 0.6 or (self.rotationN >= 0.5 and self.rotationN <= -0.5):
                self.initialvalues()
		cmd.v = 0
                cmd.omega = 0
	
        textdistance = "Num = %s, Velocity = %s, Rotation = %s, dt = %s" % (number, cmd.v, cmd.omega, dt)
        rospy.loginfo("%s" % textdistance)
        self.pub_move.publish(cmd)
        self.controltime = time

    def my_shutdown(self):
        cmd = Twist2DStamped()
	cmd.header.seq = self.number
        cmd.v = 0
        cmd.omega = 0
        self.pub_move.publish(cmd)
        rospy.sleep(1)
        self.sub_image.unregister()
        self.sub_info.unregister()
	rospy.sleep(0.5)
	self.pub.unregister()
        self.pub_move.unregister()
	self.pub_image.unregister()
	self.pub_pose.unregister()
        print("shutting down")
	rospy.sleep(0.5)
	rospy.signal_shutdown("exit")
	


if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='my_node')
    
    # keep spinning
    try:
    	rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
	
    #cv2.destroyAllWindows()
    
