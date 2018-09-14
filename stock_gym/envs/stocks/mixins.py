"""Environment for trading"""

from collections import defaultdict

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MarketEnvBase(gym.Env):
    """A mixin class for adding helpers to basic environment functionality"""
    max_observations = 256
    observation_size = 128
    total_space_size = 65536
    fee = -.001  # And/or penalty for inaction
    money = 1  # Bank
    reward_multiplier = 10
    fail_reward = 1000  # subtracted from reward on fail

    n_features = 1  # OHLCV == 5, linear values == 1
    n_actions = 3  # buy, sell, stay

    data = None

    configurables = [
        'max_observations',
        'observation_size',
        'total_space_size',
        'fee',
        'money',
        'reward_multiplier',
        'n_features',
        'n_actions',
        'data',
    ]

    position = 0  # Amount vested
    vested = 0  # Money vested

    metadata = {'render.modes': ['human']}

    idx = -1
    observed = 0

    def __init__(self, **kwargs):
        self._set_params(kwargs)

        self.action_space = self.create_action_space()
        self.observation_space = self.create_observation_space()

        self.seed()
        self.add_data(self.data)

    def _set_params(self, kwargs):
        for parm in self.configurables:
            val = kwargs.pop(parm, None)
            if val is not None:
                setattr(self, parm, val)

    def _generate_data(self):
        # Straight line at .5
        return np.full((self.n_features, self.total_space_size), .5)[0]

    def add_data(self, data=None):
        """Add data to backend"""
        if self.data is None:
            self.data = data if data is not None else self._generate_data()
        else:
            self.total_space_size = len(self.data)
            if self.observation_size > self.total_space_size:
                self.observation_size = self.total_space_size
            if self.max_observations > self.total_space_size:
                self.max_observations = self.total_space_size

    def _move_index(self):
        if self.observed == self.max_observations - 1:
            return False
        self.observed += 1
        self.idx += 1
        return True

    def get_observation(self):
        """Grab next piece of data, update index"""
        return self.data[self.idx:self.idx + self.observation_size]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_random_index(self):
        """Reset the pointer for a new run"""
        self.observed = 0
        self.idx = np.random.randint(
            self.total_space_size -
            (self.observation_size + self.max_observations - 2)
        )

    def reset(self):
        self.set_random_index()
        return self.get_observation()

    def create_action_space(self):
        """Generic discrete action space: buy, sell, stay"""
        return spaces.Discrete(self.n_actions)

    def create_observation_space(self):
        """Create a discrete space of size self.observation_size"""
        return spaces.Tuple(
            [self.create_observation_point() for ix in range(self.observation_size)]
        )

    def create_observation_point(self):
        """Create a tuple of gradients n_features wide"""
        # Range is 0-1 for normalized gradients
        return spaces.Tuple(
            [spaces.Box(low=0, high=1, shape=(1,)) for ix in range(self.n_features)]
        )


class ContinuousMarketEnvBase(MarketEnvBase):
    amount_range = 1000
    bids: dict = defaultdict(int)  # bid_price => amount

    def create_action_space(self):
        """Amount of currency to purchase at the current price"""
        return spaces.Box(
            low=-self.amount_range * self.money,
            high=self.amount_range * self.money,
            shape=(1,)
        )
