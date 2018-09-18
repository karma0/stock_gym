import pytest


TEST_INC_PARAMS = {  # Allow for 1 increment
    'max_observations': 2,
    'observation_size': 4,
    'total_space_size': 5,
    'data': [.1, .2, .3, .4, .5],
}

TEST_NOT_INC_PARAMS = {  # Don't allow for any increments
    'max_observations': 1,
    'observation_size': 5,
    'total_space_size': 5,
    'data': [.1, .2, .3, .4, .5],
}


#####
# Positive test cases
###

# RESET
def test_reset_to_zero(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    assert mkt.reset() == mkt.get_observation()
    assert mkt.idx == 0

# LAST STEP
def test_done_on_last_step(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_NOT_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0.1)
    assert done

# OBSERVATION
def test_observation_on_step_buy(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0.1)
    assert observation == mkt.data[1:]

# CALCULATE_RETURNS CALL
def test_calculate_returns_on_sell(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    price = mkt.data[-2]
    mkt.bids = {.1: 2}
    mkt.vested = .2
    mkt.position = 2
    assert round(mkt.calculate_returns(1, price), 2) == .3
    assert mkt.position == 1
    assert mkt.vested == .1

def test_calculate_returns_on_sell_eq_amt(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    price = mkt.data[-2]
    mkt.bids = {.1: 1}
    mkt.position = 1
    mkt.vested = .1
    assert round(mkt.calculate_returns(1, price), 2) == .3
    assert mkt.position == 0
    assert mkt.vested == 0

def test_calculate_returns_on_sell_hi_amt(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    price = mkt.data[-2]
    mkt.bids = {.1: 2, .2: 1}
    mkt.position = 3
    mkt.vested = .4
    assert round(mkt.calculate_returns(2, price), 2) == .5
    assert mkt.bids == {.1: 1}
    assert mkt.position == 1
    assert round(mkt.vested, 2) == .1

# REWARD
def test_reward_on_step_buy(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    price = mkt.data[-2] * .1
    (observation, reward, done, info) = mkt.step(0.1)
    calc_fee = mkt.fee * price
    assert reward == calc_fee - price

def test_reward_on_step_sell(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.bids = {.1: 1}
    mkt.position = 1
    mkt.vested = .1
    (observation, reward, done, info) = mkt.step(-1)
    assert mkt.position == 0
    assert mkt.vested == 0
    assert round(reward, 2) == 7

def test_reward_on_step_no_act(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    (observation, reward, done, info) = mkt.step(0)
    assert reward == mkt.fee

# POSITION
def test_position_on_step_buy(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 31.4
    price = mkt.data[-2]
    (observation, reward, done, info) = mkt.step(0.1)
    assert mkt.vested == price * .1
    assert mkt.position == .1

def test_position_on_step_sell(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.bids = {.1: 1}
    mkt.position = 1
    mkt.vested = .1
    price = mkt.data[-2]
    (observation, reward, done, info) = mkt.step(-0.1)
    assert mkt.bids == {.1: .9}
    assert mkt.position == .9
    assert round(mkt.money, 3) == 1.07
    assert mkt.vested == round(.9 * .1, 2)

def test_position_on_step_no_act(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 3.14
    mkt.position = 31.4
    (observation, reward, done, info) = mkt.step(0)
    assert mkt.position == 31.4

# MONEY
def test_money_on_step_buy(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 31.4
    (observation, reward, done, info) = mkt.step(0.1)
    assert mkt.money == 31.4 + reward

def test_money_on_step_sell(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.position = 31.4
    mkt.money = 3.14
    (observation, reward, done, info) = mkt.step(-0.1)
    assert mkt.money == 3.14 + (reward / mkt.reward_multiplier)

def test_money_on_step_no_act(create_i_cont_linear_market_env):
    mkt = create_i_cont_linear_market_env(TEST_INC_PARAMS)
    mkt.idx = 0
    mkt.money = 3.14
    (observation, reward, done, info) = mkt.step(0)
    assert mkt.money == 3.14 + reward
