# -*- coding: utf-8 -*-

"""Fixtures for `stock_gym` package."""

import pytest
import numpy as np
import pandas as pd

from stock_gym.envs.stocks.mixins import MarketEnvBase
from stock_gym.envs.stocks.imarket import \
        IContinuousLinearMarketEnv, IContinuousOHLCVMarketEnv, \
        IOHLCVMarketEnv, ILinearMarketEnv


#####
# Market environments
##

# Base fixture
@pytest.fixture
def create_market():
    def _create_market(market_class, kwargs):
        kwargs = {} if kwargs is None else kwargs
        return market_class(**kwargs)
    return _create_market

# MarketEnvBase
@pytest.fixture
def create_market_mixin(create_market):
    def _create_market(kwargs=None):
        return create_market(MarketEnvBase, kwargs)
    return _create_market

# ILinearMarketEnv
@pytest.fixture
def create_i_linear_market_env(create_market):
    def _create_market(kwargs=None):
        return create_market(ILinearMarketEnv, kwargs)
    return _create_market

# IContinuousLinearMarketEnv
@pytest.fixture
def create_i_cont_linear_market_env(create_market):
    def _create_market(kwargs=None):
        return create_market(IContinuousLinearMarketEnv, kwargs)
    return _create_market

# IContinuousOHLCVMarketEnv
@pytest.fixture
def create_i_cont_ohlcv_market_env(create_market):
    def _create_market(kwargs=None):
        return create_market(IContinuousOHLCVMarketEnv, kwargs)
    return _create_market

## IOHLCVMarketEnv
#@pytest.fixture
#def create_i_cont_ohlcv_market_env(create_market):
#    def _create_market(kwargs=None):
#        return create_market(IOHLCVMarketEnv, kwargs)
#    return _create_market


#####
# Misc
##

# Get flat line
@pytest.fixture
def get_flatline():
    def _create(columns, total_space_size):
        return pd.DataFrame(
            np.full((len(columns), total_space_size), .5)[0],
            columns=columns,
        )
    return _create
