from stable_baselines3 import PPO
import numpy as np
import torch.nn as nn


class Agent:
    def __init__(self):
        self.policy = None

    def load(self, path: str):
        """
        :param path: full path to a pretrained policy
        """
        self.policy = PPO.load(path)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        :param observation: numpy array
        :return: action: [throttle, turn]
        """
        action, _state = self.policy.predict(observation)
        return action
