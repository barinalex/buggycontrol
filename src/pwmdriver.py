#!/usr/bin/env python
import Adafruit_PCA9685
import time
import rospy
import numpy as np
from buggycontrol.msg import ActionsStamped
from utils import loaddefaultconfig


class PWMDriver:
    def __init__(self):
        self.config = loaddefaultconfig()
        pwm_freq = int(1. / self.config["update_period"])
        self.pulse_denominator = (1000000. / pwm_freq) / 4096.
        self.servo_ids = [0, 1] # THROTTLE IS 0, STEERING is 1
        self.throttlemin = 0.5
        self.steeringmiddle = 0.5
        print("Initializing the PWMdriver. ")
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(pwm_freq)
        self.arm_escs()
        print("Finished initializing the PWMdriver. ")
        rospy.init_node("pwmdriver")
        rospy.Subscriber("servos", ActionsStamped, self.callback)
        rospy.spin()

    def write_servos(self, vals):
        """
        :param vals: [throttle, turn]
        :return: None
        """
        for sid in self.servo_ids:
            val = np.clip(vals[sid], 0, 1)
            pulse_length = ((val + 1) * 1000) / self.pulse_denominator
            self.pwm.set_pwm(sid, 0, int(pulse_length))

    def arm_escs(self):
        """
        write lowest value to servos

        :return: None
        """
        time.sleep(0.1)
        print("Setting escs to lowest value. ")
        self.write_servos([self.throttlemin, self.steeringmiddle])
        time.sleep(0.3)

    def callback(self, msg: ActionsStamped):
        """
        :param msg: actions that are to be written to motors
        """
        self.write_servos(vals=[msg.throttle, msg.turn])


if __name__ == "__main__":
    PWMDriver()
    
