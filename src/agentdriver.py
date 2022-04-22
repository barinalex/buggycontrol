#!/usr/bin/env python3
import rospy
from buggycontrol.msg import ActionsStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped
from std_msgs.msg import Header
from threading import Lock
from agent import Agent
from queuebuffer import QueueBuffer
from waypointer import Waypointer
from state import State
import numpy as np
import tf
import tf2_ros
import time
import sys
import os


def load_raw_data(path: str) -> np.ndarray:
    """
    load data from a file. exit on fail

    :param path: path to a file
    :return: numpy array
    """
    try:
        return np.load(path)
    except Exception as e:
        sys.exit(f"ERROR WHILE LOADING DATA: {e}")


def get_static_tf(source_frame, target_frame):
    """
    :param source_frame: source frame
    :param target_frame: target frame
    :return: static transformation from source to target frame
    """
    tfbuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfbuffer)
    for i in range(100):
        print("lookup", i)
        try:
            trans = tfbuffer.lookup_transform(target_frame,
                                              source_frame,
                                              rospy.Time(0),
                                              rospy.Duration(0))
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "ros_utils tf lookup could not lookup tf: {}".format(err))
            time.sleep(0.2)
            continue


def rotate_vector_by_quat(v: Vector3, q: Quaternion):
    """
    :param v: vector
    :param q: quaternion
    :return: rotated vector by quaternion
    """
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])


class AgentDriver:
    """
    Publish actions provided by policy if allowed to act
    """
    def __init__(self):
        policypath = os.path.join("data", "policy", "ppo_tcnn_2022_04_22_11_57_47_144995.zip")
        n_wps = 10
        bufsize = 1
        pointspath = os.path.join("data", "points", "n10_wps500_smth50.npy")
        points = load_raw_data(path=pointspath)
        initvector = np.array([0, 0, 0, -1, 0])
        self.agent = Agent()
        self.agent.load(path=policypath)
        self.state = State(timestep=0.005)
        self.waypointer = Waypointer(n_wps=n_wps, points=points)
        self.actbuffer = QueueBuffer(size=bufsize, initvector=initvector)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)
        self.action = np.array([-1, 0])
        rospy.init_node("agent")
        rospy.Subscriber("joy", ActionsStamped, self.callback)
        rospy.Subscriber("/camera/odom/sample", Odometry, self.odomcallback)
        self.bl2rs = get_static_tf("odom", "camera_odom_frame")
        pub = rospy.Publisher("actions", ActionsStamped, queue_size=10)
        statepub = rospy.Publisher("imaginarystate", PoseStamped, queue_size=10)
        self.actlock = Lock()
        self.odomlock = Lock()
        self.msg = None
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            with self.actlock:
                if self.msg:
                    pub.publish(self.msg)
                    self.msg = None
                    statepub.publish(self.makepose())
            rate.sleep()

    def makepose(self) -> PoseStamped:
        pose = PoseStamped()
        pose.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
        pos = self.state.getpos()
        orn = self.state.getorn()
        pose.pose.position = Point(pos[0], pos[1], pos[2])
        pose.pose.orientation = Quaternion(x=orn[1], y=orn[2], z=orn[3], w=orn[0])
        return pose

    def updatestate(self):
        with self.odomlock:
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

    def rotate_twist(self, odom_msg):
        """
        Transform the twist in the odom message
        :param odom_msg: odometry message
        :return: transformed odometry message
        """
        odom_msg.twist.twist.linear = rotate_vector_by_quat(odom_msg.twist.twist.linear,
                                                            self.bl2rs.transform.rotation)
        odom_msg.twist.twist.angular = rotate_vector_by_quat(odom_msg.twist.twist.angular,
                                                             self.bl2rs.transform.rotation)
        return odom_msg

    def odomcallback(self, odom: Odometry):
        """
        :param odom: message containing odometry data
        """
        odom = self.rotate_twist(odom)
        with self.odomlock:
            self.lin = np.asarray([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z])
            self.ang = np.asarray([odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z])


if __name__ == "__main__":
    AgentDriver()
