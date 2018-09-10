"""Environment for trading"""

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from stock_gym.envs.stocks.imarket import IMarketEnv


class SinMarketEnv(gym.Env):
    observation_size = 25
    total_space_size = 1000
    fee = .001  # doubles as a penalty for inaction

    metadata = {'render.modes': ['human']}

    action_space = spaces.Discrete(3)
    money = 1
    position = 0
    idx = -1  # index of the start of the last observation

    def __init__(self):
        self.observation_space = spaces.Discrete(self.observation_size)
        self.data = self.gen_data()

    def set_idx_randomly(self):
        self.idx = np.random.randint(len(self.data) - self.observation_size)

    def get_observation(self):
        self.set_idx_randomly()
        return self.data[self.idx : self.idx + self.observation_size]

    def gen_data(self):
        return (1 + np.sin(np.linspace(0, 64 * np.pi, self.total_space_size))) / 2

    def step(self, action):
        assert self.action_space.contains(action)
        reward = self.fee
        if action == "buy":
            self.position = self.data[self.idx+self.observation_size]
            reward -= -1 * (self.money - self.position)
        elif action == "sell":
            reward += self.data[self.idx+self.observation_size] - self.position
            self.position = 0

        self.money += reward
        return (self.gen_data(), reward, self.money <= 0, {})

    def reset(self):
        return self.get_observation()


class LinMarketEnv(SinMarketEnv):
    def gen_data(self):
        return np.linspace(0, 1, self.total_space_size+1)[1:]


class NegLinMarketEnv(LinMarketEnv):
    def gen_data(self):
        return np.fliplr(np.atleast_2d(super().gen_data()))[0]


class OHLCVMarketEnv(IMarketEnv):
    """Market Environment
    The goal of MarketEnv is to execute trades at an effective net profit.
    """
    max_steps = 10000
    observation_size = 25
