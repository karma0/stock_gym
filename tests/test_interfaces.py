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


## RESET
def test_reset_to_zero(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    assert mkt.reset() == mkt.get_observation()
    assert mkt.idx == 0

## LAST STEP
def test_done_on_last_step(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_NOT_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0)
    assert done

## OBSERVATION
def test_accurate_observation_on_step_buy(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0)
    assert observation == mkt.data[1:]

## REWARD
def test_accurate_reward_on_step_buy(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0)
    assert reward == mkt.fee - mkt.data[-2]

def test_accurate_reward_on_step_sell(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    price = mkt.data[-2]
    mkt.position = 1
    (observation, reward, done, info) = mkt.step(1)
    assert reward == mkt.reward_multiplier * \
                        (mkt.fee + (mkt.position + (price - mkt.position)))

def test_accurate_reward_on_step_no_act(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(2)
    assert reward == mkt.fee

## POSITION
def test_accurate_position_on_step_buy(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 31.4
    price = mkt.data[-2]
    (observation, reward, done, info) = mkt.step(0)
    assert mkt.position == price

def test_accurate_position_on_step_sell(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.position = 31.4
    mkt.money = 3.14
    price = mkt.data[-2]
    (observation, reward, done, info) = mkt.step(1)
    assert mkt.position == 0

def test_accurate_position_on_step_no_act(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 3.14
    mkt.position = 31.4
    (observation, reward, done, info) = mkt.step(2)
    assert mkt.position == 31.4

## MONEY
def test_accurate_money_on_step_buy(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 31.4
    (observation, reward, done, info) = mkt.step(0)
    assert mkt.money == 31.4 + reward

def test_accurate_money_on_step_sell(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.position = 31.4
    mkt.money = 3.14
    (observation, reward, done, info) = mkt.step(1)
    assert mkt.money == 3.14 + (reward / mkt.reward_multiplier)

def test_accurate_money_on_step_no_act(create_i_linear_market_env):
    mkt = create_i_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 3.14
    (observation, reward, done, info) = mkt.step(2)
    assert mkt.money == 3.14 + reward
