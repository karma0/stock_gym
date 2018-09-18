"""Environments for trading"""

import numpy as np
import random

from stock_gym.envs.stocks.imarket import \
    IOHLCVMarketEnv, IContinuousLinearMarketEnv, ILinearMarketEnv


class SinMarketEnv(ILinearMarketEnv):
    def _generate_data(self):
        return (1 + np.sin(np.linspace(0, 64 * np.pi, self.total_space_size))) / 2


class LinMarketEnv(ILinearMarketEnv):
    def _generate_data(self):
        return np.linspace(0, 1, self.total_space_size+1)[1:]


class FlatLinMarketEnv(LinMarketEnv):
    def _generate_data(self):
        return np.full((self.n_features, self.total_space_size), .5)[0]


class NegLinMarketEnv(LinMarketEnv):
    def _generate_data(self):
        return np.fliplr(np.atleast_2d(super().gen_data()))[0]


class ContSinMarketEnv(IContinuousLinearMarketEnv):
    def _generate_data(self):
        return (1 + np.sin(np.linspace(0, 64 * np.pi, self.total_space_size))) / 2


class OHLCVMarketEnv(IOHLCVMarketEnv):
    def _generate_data(self):
        return np.fliplr(np.atleast_2d(super().gen_data()))[0]
