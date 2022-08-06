#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import threading

mutex = threading.Lock()


def joy_cb(js_data, output_data):
    output_data["lock"].acquire()
    output_data["str"] = js_data.axes[0]
    output_data["vel"] = js_data.axes[3]
    output_data["lock"].release()

    


def start():
    rospy.init_node('demo', anonymous=True)
    output_data = {"str": 0, "vel": 0, "lock":threading.Lock()}
    rate=rospy.Rate(10)
    pub = rospy.Publisher('RosAria/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/joy/", Joy, joy_cb, output_data)

    while not rospy.is_shutdown():
        output_data["lock"].acquire()

        twist = Twist()
        twist.linear.x = output_data["vel"]
        twist.angular.z = output_data["str"]
        output_data["lock"].release()
        pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    start()

