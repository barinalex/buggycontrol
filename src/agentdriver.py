#!/usr/bin/env python3
import rospy
from buggycontrol.msg import ActionsStamped
from std_msgs.msg import Header
from threading import Lock
from agent import Agent
from queuebuffer import QueueBuffer
from waypointer import Waypointer
from state import State
import numpy as np


class AgentDriver:
    """
    Publish actions provided by policy if allowed to act
    """
    def __init__(self):
        path = "ppo_tcnn_2022_04_22_11_57_47_144995.zip"
        n_wps = 10
        bufsize = 1
        points = np.random.rand((1, 500, 2))
        initvector = np.array([0, 0, 0, -1, 0])
        self.agent = Agent()
        self.agent.load(path=path)
        self.state = State(timestep=0.005)
        self.waypointer = Waypointer(n_wps=n_wps, points=points)
        self.actbuffer = QueueBuffer(size=bufsize, initvector=initvector)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)
        self.action = np.array([-1, 0])
        rospy.init_node("agent")
        rospy.Subscriber("joy", ActionsStamped, self.callback)
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

    def updatestate(self):
        self.state.set(vel=self.lin, ang=self.ang)
        self.state.update_pos()
        self.state.update_orn()

    def make_observation(self) -> np.ndarray:
        """
        :return: agent's state observation
        """
        obs = np.hstack((self.lin[:2], self.ang[2], self.action[0], self.action[1]))
        self.actbuffer.add(element=obs)
        state = self.actbuffer.get_vector()
        wps = self.waypointer.get_waypoints_vector()
        wps = np.hstack((wps, np.zeros((wps.shape[0], 1))))
        wps = self.state.toselfframe(v=wps)
        return np.hstack((state, wps[:, :2].flatten()))

    def callback(self, msg):
        """
        The button A is not pushed means agent is not allowed to drive.
        Therefore, return and don't publish actions.

        :param msg: raw input from joystick
        """
        if not msg.buttons[0]:
            return
        with self.actlock:
            self.updatestate()
            self.waypointer.update(pos=self.state.getpos()[:2])
            obs = self.make_observation()
            self.msg = ActionsStamped()
            self.action = self.agent.act(observation=obs)
            msg.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
            self.msg.throttle, self.msg.turn = self.action[0], self.action[1]
            self.msg.buttonA, self.msg.buttonB = msg.buttons[0], msg.buttons[1]


if __name__ == "__main__":
    AgentDriver()
