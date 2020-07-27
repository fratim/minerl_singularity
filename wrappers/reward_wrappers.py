import math
from gym import Wrapper

def log_reward(reward):
    reward += 1
    assert reward >= 1.0, "LogRewardWrapper is designed for rewards >= 0"
    reward = math.log2(reward)
    return reward

def sqrt_reward(reward):
    return math.sqrt(reward)

class LogRewardWrapper(Wrapper):
    """Run reward through log2 scale (if reward in [1,2], add +1 to reward) """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = log_reward(reward)
        return obs, reward, done, info

class SqrtRewardWrapper(Wrapper):
    """Run reward through sqrt """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = sqrt_reward(reward)
        return obs, reward, done, info
