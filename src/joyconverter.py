#!/usr/bin/env python3
import rospy
from buggycontrol.msg import ActionsStamped
from std_msgs.msg import Header
from sensor_msgs.msg import Joy
from threading import Lock
import numpy as np


class JoyConverter:
    """
    Receive input from joystick and convert it to
    Action messages that are published with a rate 200
    """
    def __init__(self):
        rospy.init_node("joyconverter")
        rospy.Subscriber("joy", Joy, self.callback)
        pub = rospy.Publisher("actions", ActionsStamped, queue_size=10)
        self.actlock = Lock()
        self.msg = None
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            with self.actlock:
                if self.msg:
                    pub.publish(self.msg)
                    self.msg = None
            rate.sleep()

    def callback(self, msg):
        """
        :param msg: raw input from joystick
        """
        with self.actlock:
            self.msg = ActionsStamped()
            msg.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
            self.msg.throttle, self.msg.turn = msg.axes[1], msg.axes[3]
            self.msg.buttonA, self.msg.buttonB = msg.buttons[0], msg.buttons[1]


if __name__ == "__main__":
    JoyConverter()
