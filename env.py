###example:Cartpole modify the env object for your use case

import gym

def total_reward(state, next_state ,reward_e):
    """
    compute the total reward for CartPole-v0 (reward_i is computed with the angle position range: [-41.8deg,41.8deg]

    """

    return -5000 if reward_e == 0 else 20 - abs(state[2]) - 100 * abs(next_state[2])



class Environment(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def render(self):
        self.env.render()

    def step(self,state, action):
        next_state, reward_e, done, info = self.env.step(action)
        return next_state, total_reward(state, next_state, reward_e), done, info

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
