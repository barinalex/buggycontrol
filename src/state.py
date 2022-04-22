import numpy as np
import linalg_utils as lau
import quaternion


class State:
    def __init__(self, timestep: float = 0.005):
        self.timestep = timestep
        self.pos = np.array([0, 0, 0.05])
        self.orn = quaternion.quaternion(1, 0, 0, 0)
        self.vel = np.zeros(3)
        self.ang = np.zeros(3)

    def reset(self):
        """set all to zero"""
        self.pos = np.array([0, 0, 0.05])
        self.orn = quaternion.quaternion(1, 0, 0, 0)
        self.vel = np.zeros(3)
        self.ang = np.zeros(3)

    def getpos(self):
        """
        :return: position of an agent
        """
        return np.copy(self.pos)

    def getorn(self):
        """
        :return: quaternion in a format w x y z
        """
        return quaternion.as_float_array(self.orn)

    def getlin(self):
        """
        :return: linear velocity
        """
        return np.copy(self.vel)

    def getang(self):
        """
        :return: angular velocity
        """
        return np.copy(self.ang)

    def getyaw(self):
        """
        :return: yaw computed from quaternion
        """
        _, _, yaw = self.quat2euler()
        return yaw

    def quat2euler(self):
        """
        :return: roll pitch yaw computed from quaternion
        """
        return lau.quat_to_euler(orn=self.orn)

    def vel2map(self):
        """
        :return: agent velocity in a map frame
        """
        R = quaternion.as_rotation_matrix(self.orn)
        return R @ self.vel

    def matrix2self(self):
        """
        :return: matrix for vector rotation and translation from map to agent frame
        """
        R = np.zeros((4, 4))
        R[3, 3] = 1
        R[:3, :3] = quaternion.as_rotation_matrix(self.orn.inverse())
        translation = R[:3, :3] @ self.getpos()
        R[:3, 3] = -translation
        return R

    def toselfframe(self, v: np.ndarray):
        """
        :param v: vector or vectors in a world frame
        :return: vector or vectors mapped from world to agent frame
        """
        R = self.matrix2self()
        if len(v.shape) == 1:
            v = np.hstack((v, 1))
            return R @ v
        v = np.hstack((v, np.ones((v.shape[0], 1))))
        return (R @ v.T).T

    def update_velocities(self, dvel, dang):
        """
        update velocities from t to t+1 timestep

        :param dvel: linear velocity change between t+1 and t timesteps, shape (3,)
        :param dang: angular velocity change between t+1 and t timesteps, shape (3,)
        """
        self.vel += dvel
        self.ang += dang
        # self.vel = np.clip(self.vel, -1, self.vellimit)
        # self.ang = np.clip(self.ang, -self.anglimit, self.anglimit)

    def set(self, pos=None, orn=None, vel=None, ang=None):
        """
        :param pos: position vector
        :param orn: orientation as a quaternion in a format [w x y z]
        :param vel: linear velocity
        :param ang: angular velocity
        """
        argvars = [k for k in vars().keys() if k != 'self']
        for var in argvars:
            if vars()[var] is not None:
                vars(self)[var] = np.copy(vars()[var]) if var != 'orn' \
                    else lau.array_to_quaternion(vars()[var])

    def update_pos(self):
        """
        compute new position for a timestep t+1 for 2d case
        """
        vel = self.vel2map()
        self.pos += vel * self.timestep
        self.pos[2] = 0.05

    def update_orn(self):
        """
        compute new orientation for a timestep t+1 for 2d case
        """
        delta = self.ang * self.timestep
        delta[:2] = 0
        self.orn *= quaternion.from_euler_angles(delta)

    def step(self):
        """
        update state from t to t+1 timestep
        """
        self.update_pos()
        self.update_orn()
