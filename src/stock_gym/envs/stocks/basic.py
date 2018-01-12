"""Environment for trading"""

from envs.stocks.imarket import IMarketEnv


class MarketEnv(IMarketEnv):
    """Market Environment
    The goal of MarketEnv is to execute trades at an effective net profit.
    """
    max_steps = 10000
    hist_size = 25
