"""Environment for trading"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from envs.stocks.actions import ExchangeAction
from envs.stocks.mixins import MarketMixin


class IMarketEnv(gym.Env, MarketMixin):
    """
    Base Market Environment Interface

    Actions are long, True or False, which equates to buy or sell.

    Observation space is a normalized percentage of change in either
    direction multiplied by 100 for precision in the hundredths.

    The rewards is calculated as:
    (min(action, self.number) + self.range) /
        (max(action, self.number) + self.range)

    Ideally an agent will be able to recognise the 'scent' of a higher reward
    and increase the rate in which is guesses in that direction until the
    reward reaches its maximum
    """
    max_steps = 10000
    hist_size = 25

    action = ExchangeAction()
    state = {}  # type: dict
    np_random = None
    long_start = None

    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Tuple((
            self.create_discrete_hist(),  # close
            self.create_discrete_hist(),  # volume
            self.create_discrete_hist(),  # sma
        ))

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state["closes"] = np.zeros(self.hist_size)
        self.state["volumes"] = np.zeros(self.hist_size)
        self.state["sma"] = np.zeros(self.hist_size)

        self.long_start = None
        self.total_steps = 0

        return self.action.reset()

    def _step(self, action):
        if self.action.changed(action):
            if self.action.long:  # opening up a long position
                self.long_start = self.state["closes"][-1]
            else:  # closing out of a long position
                reward = self.long_start
                self.long_start = None
            self.action.reset(action)

        done = self.total_steps > self.max_steps

        ohlcv = self.next_data()
        self.rotate(self.state["closes"], ohlcv["close"])
        self.rotate(self.state["volumes"], ohlcv["volume"])
        self.rotate(self.state["sma"], np.average(self.state["closes"]))

        return (
            self.action.long,
            reward,
            done,
            self.state
        )
