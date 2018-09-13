import pytest


TEST_INC_PARAMS = {
    'max_observations': 2,
    'observation_size': 4,
    'total_space_size': 5,
    'data': [1, 2, 3, 4, 5],
}

TEST_NOT_INC_PARAMS = {
    'max_observations': 1,
    'observation_size': 5,
    'total_space_size': 5,
    'data': [1, 2, 3, 4, 5],
}


def test_reset_to_zero(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    assert mkt.reset() == mkt.get_observation()
    assert mkt.idx == 0
