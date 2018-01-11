"""Environment for trading"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class ExchangeAction:
    """The actions of the market"""
    long = False  # By default we're not long on a position
    actions = [False, True]  # short, long

    def __init__(self, action=None):
        self.reset(action)

    def reset(self, action=None):
        if action is not None:
            assert action in self.actions
            self.long = action
        else:
            self.long = False  # default back to short
        return self.long

    def changed(self, action):
        return action is not self.long


class MarketMixin:
    """A mixin class for adding helpers to basic environment functionality"""
    data_idx = 0

    def add_data(self, data=None):
        """Add data to backend"""
        self.data = data if data is not None else self._generate_data()
        self.max_idx = len(self.data) - self.max_steps

    def next_data(self):
        """Grab next piece of data, update index"""
        self.data_idx += 1
        return self.data[self.data_idx - 1]

    def reset_idx(self):
        self.data_idx = np.random.randint(0, self.max_idx)

    def _generate_data(self):
        return self.np_random.uniform(-self.max_steps, self.max_steps)

    def create_discrete_hist(self, size=None):
        """Create a discrete space of size self.hist_size"""
        local_size = size if size is not None else self.hist_size
        return spaces.Discrete(local_size)

    def rotate(self, nparr, newitem):
        nparr = np.roll(nparr, -1)
        olditem = nparr[-1]
        nparr[-1] = newitem
        return olditem


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
