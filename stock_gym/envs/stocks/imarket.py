"""Environment for trading"""

import gym
from gym import spaces
import numpy as np

from stock_gym.envs.stocks.actions import ExchangeAction
from stock_gym.envs.stocks.mixins import \
    MarketEnvBase, ContinuousMixin, OHLCVMixin


class ILinearMarketEnv(MarketEnvBase):
    """Linear market environment
        Linear, single variable system that allows for buy, sell, stay. Doesn't
            allow for multiple consecutive buys or sells.
    """
    def step(self, action):
        assert action >= 0 and action < self.n_actions, \
                f"Invalid Action: {action} of type: {type(action)}"

        # calculate price, reward, position, and bank (money)
        price = self.data[self.idx + self.observation_size - 1]
        reward = self.fee
        if action == 0:  # buy
            if self.position > 0:  # We can only invest once at a time
                reward -= self.fail_reward
            if self.money <= 0:  # Can't buy if you have no money
                reward -= self.fail_reward
            else:
                reward -= price
            self.position += price
            self.money += reward
        elif action == 1:  # sell
            if self.position <= 0:  # Can't sell if you aren't vested
                returns = -1 * self.fail_reward
            else:
                returns = price - self.position
            reward += self.position + returns
            self.money += reward
            reward *= self.reward_multiplier
            self.position = 0  # TODO: this is problematic
        elif action == 2:  # stay
            self.money += reward

        # End if we're out of money
        done = self.money <= 0

        # Prep index for next observation or end run if we're out of time
        if not self._move_index():
            done = True

        return (
            self.get_observation(),
            reward,
            done,
            {}
        )


class IContinuousLinearMarketEnv(MarketEnvBase, ContinuousMixin):
    """Linear market environment
        Linear, single variable system that allows for buy, sell, stay.  Doesn't
            allow for multiple consecutive buys or sells.
    """
    def get_price(self):
        return self.data[self.idx + self.observation_size - 1]

    def step(self, amount):
        # calculate reward, updating price, position, and bank (money)
        reward = self.calculate_reward(amount, self.get_price())

        # End if we're out of money
        done = self.money <= 0

        # Prep index for next observation or end run if we're out of time
        if not self._move_index():
            done = True

        return (
            self.get_observation(),
            reward,
            done,
            {}
        )


class IOHLCVMarketEnv(MarketEnvBase, ContinuousMixin, OHLCVMixin):
    pass


class IContinuousOHLCVMarketEnv(MarketEnvBase, ContinuousMixin, OHLCVMixin):
    """
    Base Market Environment Interface

    Actions are represented as a continuous range of a given amount.

    Observation space is a normalized percentage of change in either
    direction multiplied by 100 for precision in the hundredths.

    The rewards is calculated as:
    (min(action, self.number) + self.range) /
        (max(action, self.number) + self.range)

    Ideally an agent will be able to recognise the 'scent' of a higher reward
    and increase the rate in which it guesses in that direction until the
    reward reaches its maximum
    """
    def get_price(self):
        return self.data.price.iloc[self.idx + self.observation_size - 1]

    def step(self, amount):
        assert amount >= -self.money and amount < self.money, \
            f"Invalid Action: {amount} for amount."

        # calculate price, reward, position, and bank (money)
        price = self.get_price()
        total_price = amount * price
        calc_fee = 0
        if total_price >= 0:  # no fee on sell
            calc_fee += self.fee * total_price  # a negative value

        reward = calc_fee
        if total_price > abs(calc_fee):  # buy
            if self.money < total_price:  # Can't buy if you have no money
                reward -= self.fail_reward
            else:
                reward -= total_price
            self.update_position(amount, price)
            self.money += reward
        elif total_price < calc_fee:  # sell
            if self.position <= amount:  # Can't sell if you aren't vested
                returns = -1 * self.fail_reward
            else:
                returns = self.update_position(amount, price)
            reward += abs(total_price) + returns
            self.money += reward
            reward *= self.reward_multiplier
        else:  # stay
            reward = self.fee
            self.money += reward

        # End if we're out of money
        done = self.money <= 0

        # Prep index for next observation or end run if we're out of time
        if not self._move_index():
            done = True

        return (
            self.get_observation(),
            reward,
            done,
            {}
        )

    #action = ExchangeAction()
    #state = {}  # type: dict
    #np_random = None
    #long_start = None

    ## Fields index
    #fidx = {
    #    "volume": -1,
    #    "close": -2,
    #    "low": -3,
    #    "high": -4,
    #    "open": -5,
    #}

    #def reset(self):
    #    self.state["opens"] = np.zeros(self.observation_size)
    #    self.state["highs"] = np.zeros(self.observation_size)
    #    self.state["lows"] = np.zeros(self.observation_size)
    #    self.state["closes"] = np.zeros(self.observation_size)
    #    self.state["volumes"] = np.zeros(self.observation_size)
    #    self.state["sma"] = np.zeros(self.observation_size)

    #    self.long_start = None
    #    self.total_steps = 0

    #    return self.action.reset()

    #def step(self, action):
    #    reward = float()
    #    if self.action.changed(action):
    #        if self.action.state.long:  # opening up a long position
    #            self.long_start = self.state["closes"][-1]
    #        if self.action.state.short:  # closing out of a long position
    #            reward = self.long_start - self.state["closes"][-1]
    #            self.long_start = None
    #        self.action.reset(action)

    #    done = self.total_steps > self.total_space_size

    #    ohlcv = self.next_data()
    #    self.rotate(self.state["opens"], ohlcv[self.fidx["open"]])
    #    self.rotate(self.state["highs"], ohlcv[self.fidx["high"]])
    #    self.rotate(self.state["lows"], ohlcv[self.fidx["low"]])
    #    self.rotate(self.state["closes"], ohlcv[self.fidx["close"]])
    #    self.rotate(self.state["volumes"], ohlcv[self.fidx["volume"]])
    #    self.rotate(self.state["sma"], np.average(self.state["closes"]))

    #    return (
    #        self.action.state,
    #        reward,
    #        done,
    #        self.state
    #    )
