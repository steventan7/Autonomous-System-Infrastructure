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
import numpy as np
from datetime import datetime
import time as t
import math
from PIL import Image

def message_organizer():
    #define global variables
    bridge = CvBridge()
    index = 0
    lasttwist = Twist()
    lasttwisttime = 0.0
    closesttwist = Twist()
    closesttwisttime = 0.0
    currentimage  = 0
    currentimagetime = 0.0
    #create numpy arrays to hold image and twist data
    imagearray = np.zeros((400,240,320,3))
    twistarray = np.zeros((400,2))
    numarray = 1
    imageadded = 0

    with rosbag.Bag('left1.bag', 'r') as outbag:
        for topic, msg, t in outbag.read_messages():
            if("_sensor_msgs__Image" in str(type(msg))): #image message
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8") #convert image msg into cv image
                currentimage = cv_image
                currentimagetime = rospy.Time.to_sec(t)
                closesttwist = lasttwist
                closesttwisttime = lasttwisttime
            if("_sensor_msgs__Image" not in str(type(msg))): #twist message
                if abs(closesttwisttime - currentimagetime) >= abs(rospy.Time.to_sec(t) - currentimagetime): 
                    #if twist message is closer, assign it to cloesettwist
                    closesttwist = msg
                    closesttwisttime = rospy.Time.to_sec(t)
                #update last msg variable
                lasttwist = msg
                lasttwisttime = rospy.Time.to_sec(t)
                #if found closest twist
                if lasttwisttime > currentimagetime and lasttwisttime > closesttwisttime:
                    #if twist is under time gap limit
                    if abs(closesttwisttime - currentimagetime) < 0.06:
                        #store image in image array
                        imagearray[index] = currentimage
                        imageadded+=1
                        #store twist linear and angular values into twist array
                        twist = np.array([closesttwist.linear.x, closesttwist.angular.z])
                        twistarray[index] = twist
                        index+=1
            
            #if numpy arrays are full, save them and then reset them
            if index > 399:
                np.save("imagearray" +str(numarray), imagearray)
                np.save("twistarray" + str(numarray), twistarray)
                numarray+=1
                imagearray = np.zeros((400,240,320,3))
                twistarray = np.zeros((400,2)) 
                index =  0     
        #save not full array at end of loop         
        np.save("imagearray" +str(numarray), imagearray)
        np.save("twistarray" + str(numarray), twistarray)

#call organization function in main
if __name__ == '__main__':
    message_organizer()

