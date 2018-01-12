"""Environment for trading"""

from gym import spaces
import numpy as np


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
        """Reset the pointer for a new run"""
        self.data_idx = np.random.randint(self.max_idx + 1)

    def _generate_data(self):
        return self.np_random.uniform(-self.max_steps, self.max_steps)

    def create_discrete_hist(self, size=None):
        """Create a discrete space of size self.hist_size"""
        local_size = size if size is not None else self.hist_size
        return spaces.Discrete(local_size)

    def rotate(self, nparr, newitem):  # pylint: disable=no-self-use
        """Rotate an item out on the left and in on the right"""
        nparr = np.roll(nparr, -1)
        olditem = nparr[-1]
        nparr[-1] = newitem
        return olditem
