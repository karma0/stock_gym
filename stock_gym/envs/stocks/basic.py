"""Environment for trading"""

import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from stock_gym.envs.stocks.imarket import IMarketEnv


class SinMarketEnv(gym.Env):
    observation_size = 25
    total_space_size = 1000
    fee = .001  # doubles as a penalty for inaction

    metadata = {'render.modes': ['human']}

    n_features = 1  # OHLCV == 5
    money = 1
    position = 0
    idx = -1  # index of the start of the last observation

    def __init__(self):
        self.observation_space = spaces.Box(
            low=0,
            high=self.total_space_size,
            shape=(self.n_features, self.observation_size)
        )
        self.action_space = spaces.Discrete(3)

        self.seed()

        self.data = self.gen_data()

    def set_idx_randomly(self):
        self.idx = np.random.randint(len(self.data) - self.observation_size)

    def get_observation(self):
        self.set_idx_randomly()
        return self.data[self.idx : self.idx + self.observation_size]

    def gen_data(self):
        return (1 + np.sin(np.linspace(0, 64 * np.pi, self.total_space_size))) / 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        return (self.get_observation(), reward, self.money <= 0, {})

    def reset(self):
        self.data = self.gen_data()
        return self.get_observation()


class LinMarketEnv(SinMarketEnv):
    def gen_data(self):
        return np.linspace(0, 1, self.total_space_size+1)[1:]


class NegLinMarketEnv(LinMarketEnv):
    def gen_data(self):
        return np.fliplr(np.atleast_2d(super().gen_data()))[0]


class FakeMarketEnv(SinMarketEnv):
    total_space_size = 10000
    volatility = .1
    start_price = .05  # start the bid generation low and fine

    def gen_element(self, old_price):
        chg = 2 * self.volatility * random.random()
        if chg > self.volatility:
            chg -= (2 * self.volatility)
        return old_price + old_price * chg

    def gen_data(self):
        elements = np.array([self.start_price])
        for a in np.range(self.total_space_size):
            elements.append(self.gen_element(elements[-1]))
        return elements


class OHLCVMarketEnv(IMarketEnv):
    """Market Environment
    The goal of MarketEnv is to execute trades at an effective net profit.
    """
    max_steps = 10000
    observation_size = 25
    n_features = 5  # OHLCV == 5
