"""Environment for trading"""

import numpy as np
import random

from stock_gym.envs.stocks.imarket import \
    IOHLCVMarketEnv, ILinearMarketEnv


class SinMarketEnv(ILinearMarketEnv):
    def gen_data(self):
        return (1 + np.sin(np.linspace(0, 64 * np.pi, self.total_space_size))) / 2


class LinMarketEnv(ILinearMarketEnv):
    def gen_data(self):
        return np.linspace(0, 1, self.total_space_size+1)[1:]


class NegLinMarketEnv(LinMarketEnv):
    def gen_data(self):
        return np.fliplr(np.atleast_2d(super().gen_data()))[0]


class FakeMarketEnv(ILinearMarketEnv):
    total_space_size = 10000
    volatility = .1
    start_price = .05  # start the bid generation low and fine

    def gen_element(self, old_price):
        chg = 2 * self.volatility * random.random()
        if chg > self.volatility:
            chg -= (2 * self.volatility)
        return old_price + old_price * chg

    def gen_data(self):
        elements = [self.start_price]
        for a in range(self.total_space_size):
            elements.append(self.gen_element(elements[-1]))
        return np.array(elements)


class OHLCVMarketEnv(IOHLCVMarketEnv):
    """Market Environment
    The goal of MarketEnv is to execute trades at an effective net profit.
    """
    max_steps = 10000
    observation_size = 25
    n_features = 5  # OHLCV == 5
