"""Environment for trading"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from stock_gym.envs.stocks.actions import ExchangeAction
from stock_gym.envs.stocks.mixins import MarketMixin


class IMarketEnv(gym.Env, MarketMixin):
    """
    Base Market Environment Interface

    Actions are long, short, or stay, all True or False.

    Observation space is a normalized percentage of change in either
    direction multiplied by 100 for precision in the hundredths.

    The rewards is calculated as:
    (min(action, self.number) + self.range) /
        (max(action, self.number) + self.range)

    Ideally an agent will be able to recognise the 'scent' of a higher reward
    and increase the rate in which it guesses in that direction until the
    reward reaches its maximum
    """
    max_steps = 10000
    hist_size = 25

    action = ExchangeAction()
    state = {}  # type: dict
    np_random = None
    long_start = None

    # Fields index
    fidx = {
        "volume": -1,
        "close": -2,
        "low": -3,
        "high": -4,
        "open": -5,
    }

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            # OHLCV
            self.create_discrete_hist(),  # open
            self.create_discrete_hist(),  # high
            self.create_discrete_hist(),  # low
            self.create_discrete_hist(),  # close
            self.create_discrete_hist(),  # volume

            # indicators
            # TODO: Add more indicators
            self.create_discrete_hist(),  # sma
        ))

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state["opens"] = np.zeros(self.hist_size)
        self.state["highs"] = np.zeros(self.hist_size)
        self.state["lows"] = np.zeros(self.hist_size)
        self.state["closes"] = np.zeros(self.hist_size)
        self.state["volumes"] = np.zeros(self.hist_size)
        self.state["sma"] = np.zeros(self.hist_size)

        self.long_start = None
        self.total_steps = 0

        return self.action.reset()

    def _step(self, action):
        reward = float()
        if self.action.changed(action):
            if self.action.state.long:  # opening up a long position
                self.long_start = self.state["closes"][-1]
            if self.action.state.short:  # closing out of a long position
                reward = self.long_start - self.state["closes"][-1]
                self.long_start = None
            self.action.reset(action)

        done = self.total_steps > self.max_steps

        ohlcv = self.next_data()
        self.rotate(self.state["opens"], ohlcv[self.fidx["open"]])
        self.rotate(self.state["highs"], ohlcv[self.fidx["high"]])
        self.rotate(self.state["lows"], ohlcv[self.fidx["low"]])
        self.rotate(self.state["closes"], ohlcv[self.fidx["close"]])
        self.rotate(self.state["volumes"], ohlcv[self.fidx["volume"]])
        self.rotate(self.state["sma"], np.average(self.state["closes"]))

        return (
            self.action.state,
            reward,
            done,
            self.state
        )
