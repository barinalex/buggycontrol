import numpy as np
from scipy.spatial.transform import Rotation as Rot
import quaternion
import math


def angle2x(vector: np.ndarray) -> float:
    """
    :param vector: shape (2,)
    :return: angle in radians
    """
    return np.arctan2(vector[1], vector[0])


def veltotraj_angle(direction: np.ndarray, vel: np.ndarray):
    """
    :param direction: two next trajectory waypoints in an agent frame
    :param vel: agent's velocity vector [Vx,Vy]
    :return: angle between the agent's velocity vector and the trajectory
    """
    carangle = np.arctan2(direction[1], direction[0])
    velangle = np.arctan2(vel[1], vel[0])
    return abs(velangle - carangle)


def get_trajectory_direction(q: quaternion, pos: np.ndarray, waypoints: np.ndarray):
    """
    :param q: robot orientation in a map frame
    :param pos: robot position in a map frame
    :param waypoints: waypoint vectors in a map frame
    :return: trajectory direction vector in the robot frame
        (difference between next two waypoint vectors)
    """
    waypoints = np.hstack((waypoints[:2], np.zeros((2, 1))))
    waypoints = torobotframe(X=waypoints, q=q, pos=pos)[:, :2]
    return waypoints[1] - waypoints[0]


def gettransformation(q: quaternion, pos: np.ndarray):
    """
    :param q: robot orientation in a map frame
    :param pos: robot position in a map frame
    :return: transformation matrix from a map to a robot frame
    """
    tf = np.zeros((4, 4))
    tf[3, 3] = 1
    tf[:3, :3] = quaternion.as_rotation_matrix(q.inverse())
    translation = tf[:3, :3] @ pos
    tf[:3, 3] = -translation
    return tf


def torobotframe(X: np.ndarray, q: quaternion, pos: np.ndarray):
    """
    :param X: vectors in a map frame
    :param q: robot orientation in a map frame
    :param pos: robot position in a map frame
    :return: vectors in a robot frame
    """
    tf = gettransformation(q=q, pos=pos)
    if len(X.shape) == 1:
        X = np.hstack((X, 1))
        return tf @ X
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return (tf @ X.T).T


def distance_point_to_line(point: np.ndarray, a: np.ndarray, b: np.ndarray):
    """return distance from the point to a line defined by two points.
    numerator is twice the area of the triangle defined by 3 given points"""
    numerator = abs((b[0] - a[0])*(a[1] - point[1]) - (a[0] - point[0])*(b[1] - a[1]))
    denominator = np.linalg.norm(b - a)
    assert denominator > 0, "DISTANCE FROM POINT TO LINE CALCULATION FAILED DUE TO DENOMINATOR = 0"
    return numerator / denominator


def projection_length(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """return distance from the point to the line as a length
    of projection to the normal vector of a line"""
    normal = get_anticlockwise_2d_normal(vec=(b - a))
    vector = point - a
    if (normal == np.zeros(*normal.shape)).all():
        return 0
    scalingfactor = (vector @ normal) / (normal @ normal)
    return np.linalg.norm(normal * scalingfactor)


def compute_determinant(matrix):
    """return determinant of a matrix"""
    return np.linalg.det(matrix)


def get_anticlockwise_2d_normal(vec):
    """return normal to a 2d vector"""
    return np.array([-vec[1], vec[0]])


def euler_from_quaternion(quat):
    """return euler"""
    return Rot.from_quat(quat).as_euler(seq='xyz')


def get_pybullet_quaternion(q):
    return np.array([q[1], q[2], q[3], q[0]])


def get_wxyz_quaternion(q):
    return np.array([q[3], q[0], q[1], q[2]])


def array_to_quaternion(q: np.ndarray):
    return quaternion.from_float_array(q)


def quat_to_euler(orn: quaternion.quaternion):
    """compute roll pitch yaw from quaternion"""
    w, x, y, z = orn.w, orn.x, orn.y, orn.z
    pitch = -math.asin(2.0 * (x * z - w * y))
    roll = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    yaw = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return roll, pitch, yaw


def compute_angular_velocity_from_orientation(orn_now: np.ndarray,
                                              orn_next: np.ndarray, time_passed: float):
    """expect orientation as a quaternion in a format w x y z"""
    q_now = array_to_quaternion(orn_now)
    q_next = array_to_quaternion(orn_next)
    q = q_next * q_now.inverse()
    roll, pitch, yaw = quat_to_euler(orn=q)
    return yaw / time_passed


def get_matrix_from_map_to_robot_frame(pos: np.ndarray, orn: quaternion.quaternion):
    """compute matrix for vector rotation and translation from map to robot frame"""
    R = np.zeros((4, 4))
    R[3, 3] = 1
    R[:3, :3] = quaternion.as_rotation_matrix(orn.inverse())
    translation = R[:3, :3] @ pos
    R[:3, 3] = -translation
    return R


def point_to_robot_frame(pos: np.ndarray, orn: quaternion.quaternion, point: np.ndarray):
    """return point in a robot frame"""
    R = get_matrix_from_map_to_robot_frame(pos=pos, orn=orn)
    return (R @ np.hstack((point, 1)))[:3]


def compute_linear_velocity_from_position(pos_now: np.ndarray, pos_next: np.ndarray,
                                          orn_now: np.ndarray, time_passed: float):
    """expect orientation as a quaternion in a format w x y z"""
    q_now = array_to_quaternion(orn_now)
    traveled_distance = point_to_robot_frame(pos=pos_now, orn=q_now, point=pos_next)
    return traveled_distance / time_passed


if __name__ == "__main__":
    a = np.random.rand(2)
    b = np.random.rand(2)
    p = np.random.rand(2)
    print(distance_point_to_line(p, a, b), projection_length(p, a, b))
