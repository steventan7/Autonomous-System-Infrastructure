#!/usr/bin/env python
from __future__ import print_function
import rosbag
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

def image_converter():
    bridge = CvBridge()
    with rosbag.Bag('autobag4.bag', 'r') as outbag:
        for topic, msg, t in outbag.read_messages():
            if("_sensor_msgs__Image" in str(type(msg))):    
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imshow("Image window", cv_image)
                cv2.waitKey(3)
   
if __name__ == '__main__':
    image_converter()
    
