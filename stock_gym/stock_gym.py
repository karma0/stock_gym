# -*- coding: utf-8 -*-

import gym
from stock_gym.envs import stocks


"""Main module."""
def main():
    #env = gym.make('LinMarketEnv-v0')
    #env = gym.make('NegLinMarketEnv-v0')
    env = gym.make('SinMarketEnv-v0')

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            #env.render()
            #print(f"Observation {observation}")
            action = env.action_space.sample()
            observation, reward, done_, info = env.step(action)
            if done_:
                print(f"Episode finished after {t+1} timesteps")
            else:
                print(f"Reward: {reward}")
