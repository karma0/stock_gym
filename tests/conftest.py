# -*- coding: utf-8 -*-

"""Fixtures for `stock_gym` package."""

import pytest
import numpy as np

from stock_gym.envs.stocks.mixins import MarketMixin
from stock_gym.envs.stocks.imarket import ILinearMarketEnv


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

@pytest.fixture
def create_market():
    def _create_market(market_class, kwargs):
        kwargs = {} if kwargs is None else kwargs
        return market_class(**kwargs)
    return _create_market

@pytest.fixture
def create_market_mixin(create_market):
    def _create_market(kwargs=None):
        return create_market(MarketMixin, kwargs)
    return _create_market

@pytest.fixture
def create_i_linear_market_env(create_market):
    def _create_market(kwargs=None):
        return create_market(ILinearMarketEnv, kwargs)
    return _create_market


@pytest.fixture
def get_flatline():
    def _create(n_features, total_space_size):
        return np.full((n_features, total_space_size), .5)[0].all()
    return _create
