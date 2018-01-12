"""Initialize the env.stocks module"""

from gym.envs.registration import registry, register, make, spec

# Public classes
from stock_gym.envs.stocks.basic import MarketEnv
from stock_gym.envs.stocks.imarket import IMarketEnv


register(
    id='MarketEnv-v0',
    entry_point='stock_gym.envs.stocks:MarketEnv',
    )
