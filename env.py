###example:Cartpole modify the env object for your use case

import gym
import numpy as np

def total_reward_cartpole(state, next_state ,reward_e):
    """
    compute the total reward for CartPole-v0 (reward_i is computed with the angle position range: [-41.8deg,41.8deg]

    """

    return -5000 if reward_e == 0 else 20 - abs(state[2]) - 100 * abs(next_state[2])

def total_reward_mountain(state, next_state, reward_e):
    return reward_e

def mountaincargoal(state):
    raise NotImplementedError

def cartpole_goal(times_alive):
    if np.mean(times_alive) >= 200:
        return True
    else:
        return False


class Environment(object):
    def __init__(self, env_name):
        print(env_name)
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        if self.env_name == 'CartPole-v0':
            self.total_reward = total_reward_cartpole
            self.goal = cartpole_goal
        elif self.env_name == 'MountainCar-v0':
            self.total_reward = total_reward_mountain


    def render(self):
        self.env.render()

    def step(self,state, action):
        next_state, reward_e, done, info = self.env.step(action)
        return next_state, self.total_reward(state, next_state, reward_e), done, info

    def initialize(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def get_spaces(self):
        """
        :return:
        action_space, observation_space

        """
        return self.env.action_space.n, self.env.observation_space.shape
