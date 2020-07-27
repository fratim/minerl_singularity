import numpy as np
import cv2

from gym import spaces, Wrapper


def resize_image(img, width_and_height):
    new_img = cv2.resize(img, width_and_height)
    return new_img


class MineRLPOVResizeWrapper(Wrapper):
    """
    Simple MineRL specific POV-image
    resizer
    """

    def __init__(self, env, target_shape=64):
        """
        target_shape: Image dimension per axis
        """
        super().__init__(env)
        self.target_shape = target_shape

        new_spaces = self.env.observation_space
        new_spaces.spaces["pov"] = spaces.Box(low=0, high=255, shape=(self.target_shape, self.target_shape, 3), dtype=np.uint8)
        self.observation_space = new_spaces

    def _check_and_resize(self, obs):
        if self.target_shape != 64:
            # Do simple copy to avoid modifying obs in place
            obs = dict(obs.items())
            obs["pov"] = resize_image(obs["pov"], (self.target_shape, self.target_shape))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._check_and_resize(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._check_and_resize(obs)
        return obs
