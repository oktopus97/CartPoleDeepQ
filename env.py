###example:Cartpole modify the env object for your use case

import gym

def total_reward(state,reward_e, timestep):
    """
    compute the total reward for CartPole-v0 (reward_i is computed with the angle position range: [-41.8deg,41.8deg]
    if the agent lives more then it will receive more rewards
    """

    return reward_e + timestep/20



class Environment(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def render(self):
        self.env.render()

    def step(self, action, timestep):
        state, reward_e, done, info = self.env.step(action)
        return state, total_reward(state, reward_e, timestep), done, info

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
