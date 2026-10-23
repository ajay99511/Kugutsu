"""MuJoCo scenes (assets/*.xml) and Gymnasium wrappers."""

import gymnasium as gym


class RobotEnv(gym.Env):
    """Placeholder Gymnasium wrapper; load an XML from assets/ with MuJoCo."""

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
