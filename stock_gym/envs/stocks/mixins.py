"""Environment for trading"""

from gym import spaces
from gym.utils import seeding

import numpy as np


class MarketMixin:
    """A mixin class for adding helpers to basic environment functionality"""
    max_observations = 256
    observation_size = 128
    total_space_size = 65536
    fee = -.001  # And/or penalty for inaction
    money = 1  # Bank
    reward_multiplier = 10
    reward_failure = 1000  # subtracted from reward on fail

    n_features = 1  # OHLCV == 5, linear values == 1
    n_actions = 3  # buy, sell, stay

    data = None

    changeable_params = [
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

    metadata = {'render.modes': ['human']}

    idx = -1
    observed = 0
    position = 0  # Money vested

    def __init__(self, **kwargs):
        self._set_params(kwargs)

        self.action_space = self.create_discrete_actions()
        self.observation_space = self.create_box_hist()

        self.seed()
        self.add_data(self.data)

    def _set_params(self, kwargs):
        for parm in self.changeable_params:
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

    def create_discrete_actions(self):
        return spaces.Discrete(self.n_actions)

    def create_box_hist(self):
        """Create a discrete space of size self.observation_size"""
        return spaces.Box(
            low=0,
            high=self.total_space_size,
            shape=(self.n_features, self.observation_size),
        )

    def rotate(self, nparr, newitem):  # pylint: disable=no-self-use
        """Rotate an item out on the left and in on the right"""
        nparr = np.roll(nparr, -1)
        olditem = nparr[-1]
        nparr[-1] = newitem
        return olditem
