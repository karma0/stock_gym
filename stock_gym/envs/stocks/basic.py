"""Environments for trading"""

import numpy as np
import random

from stock_gym.envs.stocks.imarket import \
    IOHLCVMarketEnv, IContinuousLinearMarketEnv, ILinearMarketEnv, \
    IContinuousOHLCVMarketEnv


class SinMarketEnv(ILinearMarketEnv):
    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        return (1 + np.sin(np.linspace(0, 64 * np.pi, length))) / 2


class LinMarketEnv(ILinearMarketEnv):
    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        return np.linspace(0, 1, length+1)[1:]


class FlatLinMarketEnv(LinMarketEnv):
    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        return np.full((self.n_features, length), .5)[0]


class NegLinMarketEnv(LinMarketEnv):
    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        return np.fliplr(np.atleast_2d(super().gen_data(length=length)))[0]


class ContSinMarketEnv(IContinuousLinearMarketEnv):
    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        return (1 + np.sin(np.linspace(0, 64 * np.pi, length))) / 2


#class OHLCVMarketEnv(IOHLCVMarketEnv):
#    def _generate_data(self, length=None):
#        return np.fliplr(np.atleast_2d(super().gen_data(length=length)))[0]


class OHLCVMarketEnv(IContinuousOHLCVMarketEnv):
    pass
