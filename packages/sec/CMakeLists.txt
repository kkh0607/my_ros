cmake_minimum_required(VERSION 2.8.3)
project(sec)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  duckietown_msgs
  rospy
)

catkin_package()
