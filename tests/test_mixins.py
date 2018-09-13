import pytest


TEST_PARAMS = {
    'max_observations': 123,
    'observation_size': 12,
    'total_space_size': 12345,
    'fee': .123,
    'money': 1234,
    'reward_multiplier': 12,
    'n_features': 12,
    'n_actions': 12,
}


def test_set_parameters(create_market_mixin):
    mkt = create_market_mixin(TEST_PARAMS)
    assert mkt.max_observations == 123
    assert mkt.observation_size == 12
    assert mkt.total_space_size == 12345
    assert mkt.fee == .123
    assert mkt.money == 1234
    assert mkt.reward_multiplier == 12
    assert mkt.n_features == 12
    assert mkt.n_actions == 12
    assert len(mkt.data) == 12345

def test_set_data(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({'data': data})

    assert len(mkt.data) == len(data)
    assert mkt.data == data
    assert mkt.total_space_size == len(data)
    assert mkt.observation_size == len(data)
    assert mkt.max_observations == len(data)

def test_generate_data(create_market_mixin, get_flatline):
    mkt = create_market_mixin(TEST_PARAMS)
    # Should be a straight line at .5
    assert mkt.data.all() == get_flatline(12, 12345)

def test_get_random_index(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({
        'max_observations': 1,
        'observation_size': len(data),
        'total_space_size': len(data),
        'data': data,
    })
    assert mkt.idx == -1
    mkt.set_random_index()
    assert mkt.idx == 0
    assert mkt.idx < (mkt.total_space_size -
                      (mkt.observation_size + mkt.max_observations - 2))

def test_move_index(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({
        'max_observations': 2,
        'observation_size': len(data) - 1,
        'total_space_size': len(data),
        'data': data,
    })
    mkt.idx = 0
    assert mkt._move_index()
    assert mkt.idx > 0
    assert mkt.idx <= (mkt.total_space_size -
                      (mkt.observation_size + mkt.max_observations - 2))

def test_move_index_past_end(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({
        'max_observations': 2,
        'observation_size': len(data) - 1,
        'total_space_size': len(data),
        'data': data,
    })
    mkt.idx = 0
    assert mkt._move_index()
    assert not mkt._move_index()

def test_move_index_past_max(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({
        'max_observations': 1,
        'observation_size': len(data) - 1,
        'total_space_size': len(data),
        'data': data,
    })
    mkt.idx = 0
    assert not mkt._move_index()

def test_get_observation(create_market_mixin):
    data = [1, 2, 3, 4, 5]
    mkt = create_market_mixin({
        'max_observations': 2,
        'observation_size': len(data) - 1,
        'total_space_size': len(data),
        'data': data,
    })
    mkt.idx = 0
    assert mkt._move_index()
    assert len(mkt.get_observation()) == mkt.observation_size
    assert mkt.get_observation() == mkt.data[1:]
