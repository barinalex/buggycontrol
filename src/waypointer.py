import numpy as np
import linalg_utils as lau
from multiprocessing import Lock
mutex = Lock()


class Waypointer:
    def __init__(self, n_wps: int, points: np.ndarray, continuous: bool = True, random: bool = False):
        """
        :param n_wps: number of waypoints per step
        :param points: shape (m, k, 2) m trajectories, k waypoints per trajectory, 2 dimensions
        :param continuous: if True join trajectory to the end of previous when necessary
        :param random: if true after reset new trajectory is chosen randomly,
            if false go sequentially over the list of trajectories
        """
        self.nextwp = 0
        self.next = 0
        self.n_wps = n_wps
        self.points = points
        self.continuous = continuous
        self.random = random
        self.waypoints = {}
        self.mjviz = None
        self.reset()

    def reset(self):
        """
        next trajectory to waypoints
        """
        with mutex:
            self.nextwp = 0
            self.next = (self.next + 1) % self.points.shape[0]
            self.waypoints = self.points2trajectory()

    def gettrajectoryindex(self):
        """
        :return: index of points to define new trajectory.
            random or sequential choice
        """
        if self.random:
            return np.random.randint(low=0, high=self.points.shape[0])
        return self.next

    def points2trajectory(self) -> dict:
        """
        :return: dictionary containing waypoints with its state (visited/not)
        """
        points = self.points[self.gettrajectoryindex()]
        visited = np.zeros(points.shape[0], dtype=bool)
        return {"points": points, "visited": visited}

    def stack_waypoints(self, wps1: dict, wps2: dict) -> dict:
        """
        :param wps1: previous waypoints
        :param wps2: next waypoints
        :return: concatenated waypoints
        """
        wps2['points'] += wps1['points'][-1]
        for key in wps1.keys():
            wps1[key] = np.concatenate((wps1[key], wps2[key]))
        return wps1

    def next_unvisited_point(self) -> np.ndarray:
        """
        :return: next unvisited points, or if all are visited return None
        """
        return np.copy(self.waypoints["points"][self.nextwp])

    def get_last_visited_point(self) -> np.ndarray:
        """
        :return: last visited point
        """
        return np.copy(self.waypoints["points"][self.nextwp - 1]) if self.nextwp > 1 else np.zeros(2)

    def get_waypoints_vector(self) -> np.ndarray:
        """
        :return: vector with waypoints
        """
        return np.copy(self.waypoints["points"][self.nextwp: self.nextwp + self.n_wps])

    def nextvisited(self, pos) -> bool:
        """
        :param pos: position of an agent. shape (2,)
        :return: true if next waypoint was visited
        """
        next = self.next_unvisited_point()
        prev = self.get_last_visited_point()
        normalvec = lau.get_anticlockwise_2d_normal(vec=(next - prev))
        agentvec = next - pos
        matrix = np.stack((normalvec, agentvec), axis=1)
        return lau.compute_determinant(matrix=matrix) > 0

    def distance_to_trajectory(self, pos) -> float:
        """
        :param pos: position of an agent. shape (2,)
        :return: distance from the current position point to the trajectory
        """
        next = self.next_unvisited_point()
        prev = self.get_last_visited_point()
        return lau.projection_length(point=pos, a=prev, b=next)

    def is_end(self) -> bool:
        """
        :return: True if trajectory has less than n waypoints left
        """
        nwps = self.waypoints["points"].shape[0]
        return nwps - self.n_wps <= self.nextwp

    def continue_trajectory(self, force=False):
        """
        stack new (loaded or random) trajectory to the end of current trajectory
        :param force: if true continue trajectory regardless of conditions
        """
        if self.continuous and (force or self.needtocontinue()):
            self.waypoints = self.stack_waypoints(self.waypoints, self.waypoints)

    def needtocontinue(self):
        """
        :return: true if amount of unvisited points is < number of waypoints
        """
        return len(self.waypoints["points"]) - self.nextwp < self.n_wps

    def update(self, pos):
        """
        :param pos: position of an agent. shape (2,)
        """
        if not self.nextvisited(pos):
            return False
        self.mark_visited()
        self.continue_trajectory()
        self.visualize()
        return True

    def mark_visited(self):
        """
        mark waypoints as visited and iterate to the next
        """
        self.waypoints["visited"][self.nextwp] = True
        self.nextwp += 1

    def visualize(self):
        """
        update visual state of waypoints
        """
        if self.mjviz is None:
            return
        try:
            wppos = self.waypoints["points"][self.nextwp + self.mjviz.n_wps - 1]
            self.mjviz.movewaypoint(pos=wppos)
        except Exception as e:
            print(f"EXCEPTION {e} OCCURRED")
