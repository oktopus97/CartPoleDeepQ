import random
import math
import numpy as np

from prep import prep_mem_batch, prep_exploration,prep_exploitation,prep_q,device

from hyper_params import *
from env import Environment
from network import DeepQNetwork
from replaymemory import ReplayMemory
import torch.autograd as autograd
import torch.optim as optim
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
import torch

import string


class GymAgent(object):
    """
    baseclass for agents using the OpenAi Gym environments
    """

    agent_no = 0
    goal = None

    def __init__(self, env, mode, tensorboard_writer=None):
        GymAgent.agent_no += 1
        if tensorboard_writer:
            self.writer = tensorboard_writer

        self.env = env
        self.action_space, self.obs_space = self.env.get_spaces()
        self.mode = mode


        if mode == 'train':
            self.no_training_steps = 0

class DQNAgent(GymAgent):
    """
    an agent for running the DQN algorithm (Minh et al 2013)
    """

    def __init__(self, env, mode, pre_trained_model, tensorboard_writer=None):
        super(DQNAgent, self).__init__(env, mode, tensorboard_writer)
        self.agent_name = 'DQN' + str(self.agent_no)
        self.memory = ReplayMemory()

        self.network = DeepQNetwork(self.obs_space[0],self.action_space)



        if self.mode == 'play':
            self.network.load_params(pre_trained_model)
            self.network.eval()

        elif self.mode == 'train':

            self.eval_network = DeepQNetwork(self.obs_space[0], self.action_space)
            self.eval_network.eval()

            if pre_trained_model:
                self.eval_network.load_params(pre_trained_model)

            self.optimizer = optim.RMSprop(self.network.parameters(), lr=LR)
            self.loss_func = SmoothL1Loss()
        else:
            raise ValueError('Please set a valid mode for the agent (play or train)')

    def interact(self, state, action):
        """
        returns:
        state, reward, done, info
        """
        return self.env.step(action, state)

    def select_action(self,state):
        if self.mode == 'play':
            return self.network(prep_exploitation(state)).max(1)[1].view(1, 1)
        ##epsilon greedy policy
        eps_threshold = EPS_START * EPS_DECAY ** self.no_training_steps if EPS_DECAY > EPS_END else EPS_END

        self.no_training_steps += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.network(prep_exploitation(state)).max(1)[1].view(1, 1)
        else:
            return prep_exploration(self.action_space)
    def optimize(self):
        sum_loss = 0

        if len(self.memory) < BATCH_SIZE:
            batch_size = len(self.memory)
        else:
            batch_size = BATCH_SIZE


        s, a, _s, r = prep_mem_batch(self.memory.sample(batch_size))

        non_final_next = torch.cat([sa for sa in _s if sa is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, _s)))
        state_action_values = self.network(s).gather(1, a.long().unsqueeze(1))

        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.eval_network(non_final_next).detach().max(1)[0]

        expected_q = prep_q(next_state_values, r)
        loss = self.loss_func(state_action_values, expected_q.unsqueeze(1))


        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()


    def train(self, num_episodes, render=False, lr_decay=False):

        end_state = np.zeros(self.obs_space)
        state = end_state

        for episode in range(1, num_episodes + 1):
            done = False
            timesteps = 0
            rewards = []
            sum_rewards = []
            loss = 0
            times_alive = []

            while not done:
                if state is end_state:
                    state = self.env.initialize()

                if render: self.env.render()
                action = self.select_action(state)
                _state, reward, done, _ = self.interact(action.item(), state)
                rewards.append(reward)

                timesteps += 1

                if done:
                    _state = end_state

                    sum_reward = np.sum(rewards)
                    sum_rewards.append(sum_reward)

                    mean_loss = loss / timesteps
                    times_alive.append(timesteps)
                    timesteps = 0

                    if self.writer:
                        self.writer.add_scalar(self.agent_name + 'duration of episode', timesteps, episode)
                        self.writer.add_scalar(self.agent_name + 'mean reward of episode', sum_reward, episode)
                        self.writer.add_scalar(self.agent_name + 'mean loss of episode', mean_loss, episode)

                self.memory.push(state, action, _state if _state is not None else end_state, reward)

                state = _state
                episode_loss = self.optimize()
                loss += episode_loss

            if lr_decay:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + (episode / LR_DECAY))


            if episode % TARGET_UPDATE == 0:
                if self.env.goal(times_alive):
                    print('goal reached your computer is smart :)')
                    self.eval_network.save_params(self.agent_name, self.env.env_name)
                    break
                else:
                    times_alive = []

                self.eval_network.update_params(self.network)
                print('episode ', episode, 'loss ', mean_loss, 'reward ', np.mean(sum_rewards))
                #add your custom goals




    def play(self, num_episodes):
        for episode in range(1, num_episodes + 1):
            done = False
            state = self.env.initialize()
            while not done:
                self.env.render()
                action = self.select_action(state)
                _state, reward, done, _ = self.interact(action.item(), state)
                if done:
                    state = self.env.initialize()
