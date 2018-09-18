"""Environment for trading"""

from collections import defaultdict
import random

import numpy as np
import pandas as pd

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

    # Market generation metrics
    volitility = .1  # volitility for generated data
    start_price = .1

    columns = ['price']  # DataFrame columns
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
        'columns',
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

    def _gen_element(self, last_price):
        change = 2 * self.volitility * random.random()
        if change > self.volitility:
            change -= (2 * self.volitility)
        return last_price + last_price * change

    def _generate_data(self, length=None):
        length = self.total_space_size if length is None else length
        # Return a straight line at .5
        return pd.DataFrame(
            np.full((self.n_features, length), .5)[0],
            columns=self.columns,
        )

    def add_data(self, data=None, length=None):
        """Add data to backend"""
        if self.data is None:  # allows for init override of data
            self.data = self._generate_data(length) \
                        if data is None \
                        else pd.DataFrame(data, columns=self.columns)
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
            [spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                for ix in range(self.n_features)]
        )


class OHLCVMixin:
    """Provides utility functions and boiler configuration around OHLCV"""
    n_features = 5  # OHLCV == 5, linear values == 1
    columns = ['price', 'quantity']
    time_start = '1/1/2018'
    time_end = '1/2/2018'
    time_freq = 'S'  # Nanosecond-level granularity
    ohclv_freq = '30Sec'

    # Add this many samples for downsample to OHLCV. This should be the
    #  average number of order executions per generated OHLCV period.
    samplesize = .75

    # Base volume by which to generate random movement
    volume_base = .1

    lastrow = None
    generated_row_count = 0

    def convert_to_ohlcv(self):
        self.raw_data = self.data.copy()
        ohlc = self.data['price'].resample(self.ohclv_freq)
        volume = self.data['quantity'].resample(self.ohlcv_freq).sum().fillna(0)

        ohlcv = pd.concat([ohlc, volume], axis=1)
        ohlcv.columns.values[-1] = 'volume'

        self.lastrow = ohlcv.iloc[0].copy()

        def update_nan(row):
            """Forward fill NaN in OHLC fields"""
            if row.volume == 0:
                row.open = self.lastrow.close
                row.high = self.lastrow.close
                row.low = self.lastrow.close
                row.close = self.lastrow.close
            self.lastrow = row.copy()
            return row

        self.data = ohlcv.apply(update_nan, axis=1)
        self.lastrow = None

    def add_time_index(self, length=None):
        length = self.generated_row_count if length is None else length
        dates = pd.DataFrame(
            {
                'timestamp': pd.date_range(
                    start=self.time_start,
                    end=self.time_end,
                    freq=self.time_freq,
                ),
            },
            index='timestamp',
        ).sample(self.generated_row_count)

        self.data = pd.concat([dates, self.data], axis=1)
        self.data.set_index('timestamp')

    def _generate_data(self, length=None):
        length = self.generated_row_count if length is None else length
        prices = [self.start_price]
        for a in range(length):
            new_price = self._gen_element(prices[-1])
            amount = random.random() * (new_price - prices[-1])
            prices.append([new_price, amount])
        return pd.DataFrame.from_records(prices, columns=self.columns)

    def add_data(self, data=None, length=None):
        """Add data to backend"""
        if data is None:
            self.generated_row_count = \
                round((1 + self.samplesize) * self.total_space_size)
        else:
            self.generated_row_count = len(data)

        length = self.generated_row_count if length is None else length
        #self.data = self._generate_data(length)
        super().add_data(data=data, length=self.generated_row_count)

        self.add_time_index()
        self.convert_to_ohlcv()


class ContinuousMixin:
    """Provides a continuous action interface"""
    amount_range = 1000
    bids: dict = defaultdict(int)  # bid_price => amount
    money: int

    position = 0  # Amount vested
    vested = 0  # Money vested

    def create_action_space(self):
        """Amount of currency to purchase at the current price"""
        return spaces.Box(
            low=-self.amount_range * self.money,
            high=self.amount_range * self.money,
            shape=(1,),
            dtype=np.float32,
        )

    def calculate_reward(self, amount, price):
        total_price = amount * price
        if total_price > 0:  # buy
            return self.go_long(amount, price)
        elif total_price < 0:  # sell
            return self.short(abs(amount), price)
        else:  # stay
            return self.stay()

    def go_long(self, amount, price):
        total_price = amount * price
        reward = self.fee * total_price

        if self.money < total_price:  # Can't buy if you have no money
            reward -= self.fail_reward
        else:
            reward -= total_price

        self.bids[price] += amount
        self.position += amount
        self.vested += total_price
        self.money += reward
        return reward

    def short(self, amount, price):
        total_price = amount * price
        reward = self.fee * total_price
        returns = self.calculate_returns(amount, price)
        reward += total_price + returns
        self.money += reward
        return reward * self.reward_multiplier

    def stay(self):
        self.money += self.fee
        return self.fee

    def calculate_returns(self, amount, price):
        if self.position < amount:
            return -1 * self.fail_reward

        selling_vested = 0
        amt = amount
        for _bid, _amt in reversed(sorted(self.bids.items())):
            if amt > 0:
                if amt >= _amt:
                    selling_vested += _bid * _amt
                    self.bids.pop(_bid)
                    self.position -= _amt
                    amt -= _amt
                else:
                    selling_vested += _bid * amt
                    self.bids[_bid] -= amt
                    self.position -= amt
                    amt = 0
            else:
                break

        self.vested -= selling_vested
        return amount * price - selling_vested
