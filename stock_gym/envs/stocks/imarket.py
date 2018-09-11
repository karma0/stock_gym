"""Environment for trading"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from stock_gym.envs.stocks.actions import ExchangeAction
from stock_gym.envs.stocks.mixins import MarketMixin


class ILinearMarketEnv(gym.Env):
    max_observations = 128
    observation_size = 64
    total_space_size = 8192
    fee = .001  # And/or penalty for inaction

    metadata = {'render.modes': ['human']}

    n_features = 1  # OHLCV == 5
    n_actions = 3  # buy, sell, stay
    money = 1  # Bank
    position = 0  # Set to bid price at beginning
    idx = -1  # index of the start of the last observation

    def __init__(self):
        self.observation_space = spaces.Box(
            low=0,
            high=self.total_space_size,
            shape=(self.n_features, self.observation_size),
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self.seed()

        self.data = self.gen_data()

    def move_index(self):
        self.idx += 1

    def set_random_index(self):
        self.idx = np.random.randint(
            len(self.data) -
            self.observation_size -
            self.max_observations
        )

    def get_observation(self):
        return self.data[self.idx:self.idx + self.observation_size]

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
        self.move_index()
        return (self.get_observation(), reward, self.money <= 0, {})

    def reset(self):
        self.set_random_index()
        self.data = self.gen_data()
        return self.get_observation()


class IOHLCVMarketEnv(gym.Env, MarketMixin):
    """
    Base Market Environment Interface

    Actions are "buy", "stay", "sell".

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
    observation_size = 25

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

    def __init__(self, **kwargs):
        # Setup configuration overrides
        for kw in ['max_steps', 'observation_size']:
            val = kwargs.pop(kw, None)
            if val is not None:
                setattr(self, kw, val)

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
        self.state["opens"] = np.zeros(self.observation_size)
        self.state["highs"] = np.zeros(self.observation_size)
        self.state["lows"] = np.zeros(self.observation_size)
        self.state["closes"] = np.zeros(self.observation_size)
        self.state["volumes"] = np.zeros(self.observation_size)
        self.state["sma"] = np.zeros(self.observation_size)

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
