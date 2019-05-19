import random
import math
import numpy as np

from prep import prep_mem_batch, prep_exploration,prep_exploitation,prep_q,device

from hyper_params import *
from env import Environment
from network import DeepQNetwork

from replaymemory import Memory
import torch.autograd as autograd
import torch.optim as optim
from torch.nn import MSELoss
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

class Agent(object):
    def __init__(self):
        
        self.memory = Memory()

        self.env = Environment()
        self.action_space, self.obs_space = self.env.get_spaces()
        self.network = DeepQNetwork(self.obs_space[0],self.action_space)
        self.eval_network = DeepQNetwork(self.obs_space[0], self.action_space)

        self.no_training_steps = 0

        self.optimizer = optim.Adam(self.network.parameters(),lr=LR)
        self.loss_func = MSELoss()


    

    def interact(self,action):
        """
        returns:
        state, reward, done, info
        """
        return self.env.step(action)

    def select_action(self,state):
        ##epsilon greedy policy
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-self.no_training_steps/EPS_DECAY)

        self.no_training_steps += 1
        
        if random.random() > eps_threshold:
            return self.eval_network(prep_exploitation(state)).max(1)[1].view(1, 1)
        else:
            return prep_exploration(self.action_space)
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        s, a, _s, r = prep_mem_batch(self.memory.sample())
        
        state_action_values = self.network(s).gather(1, a.long().unsqueeze(1))

        max_next_state_values = self.eval_network(_s).max(1)[0].detach()


        expected_q = prep_q(max_next_state_values,r)

        expected_q.requires_grad = True
        
        loss = self.loss_func(state_action_values.squeeze(),expected_q)
        print('lolo')

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        print('lolo')
        


    def train(self, num_episodes):
        try:
            self.eval_network.load_params()
        except:
            print('no params found')

        end_state = np.array([0,0,0,0])
        state = end_state
        timesteps = 0
        timesalive = []

        for episode in range(num_episodes):

            if state is end_state:
                state = self.env.initialize()

            action = self.select_action(state)
            _state, reward, done, _ = self.interact(action.item())
            timesteps += 1


            if done:
                _state = end_state
                print('alive for {} timesteps'.format(timesteps))
                timesalive.append(timesteps)
                timesteps = 0
            self.memory.push(state, action, _state if _state is not None else end_state, reward)

            state = _state


            self.optimize()

            if episode % TARGET_UPDATE == 0:
                self.eval_network.update_params(self.network)

        plt.plot(timesalive)
        plt.show()


agent_smith = Agent()
print('agent initiated')
agent_smith.train(10000)
