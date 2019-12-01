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
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

losses = []
timesalive = []

class Agent(object):
    def __init__(self):

        self.memory = Memory()

        self.env = Environment()
        self.action_space, self.obs_space = self.env.get_spaces()

        self.network = DeepQNetwork(self.obs_space[0],self.action_space)

        self.eval_network = DeepQNetwork(self.obs_space[0], self.action_space)
        self.eval_network.eval()

        self.no_training_steps = 0

        self.optimizer = optim.RMSprop(self.network.parameters(), lr=LR)
        self.loss_func = SmoothL1Loss()


    def interact(self, state, action):
        """
        returns:
        state, reward, done, info
        """
        return self.env.step(action, state)

    def select_action(self,state):
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
        """
            target = r
            if r != 0:
                target = prep_q(r, self.network(_s.cuda()).max(1)[0])

            netwotk_output = self.network(s.cuda()).squeeze()
            target_f = netwotk_output.clone().detach()

            target_f[int(a.item())]= target

            loss = self.loss_func(target_f, netwotk_output)


        """
        
    def train(self, num_episodes):

        end_state = np.array([0,0,0,0])
        state = end_state



        for episode in range(1, num_episodes + 1):
            done = False
            timesteps = 0
            rewards = []
            sum_rewards = []
            loss = 0

            while not done:
                if state is end_state:
                    state = self.env.initialize()

                self.env.render()
                action = self.select_action(state)
                _state, reward, done, _ = self.interact(action.item(), state)
                rewards.append(reward)

                timesteps += 1
                if done:
                    _state = end_state

                    sum_reward = np.sum(rewards)
                    sum_rewards.append(sum_reward)

                    mean_loss = loss / timesteps

                    writer.add_scalar('duration of episode', timesteps, episode)
                    writer.add_scalar('mean reward of episode', sum_reward, episode)
                    writer.add_scalar('mean loss of episode', mean_loss, episode)

                    timesalive.append(timesteps)
                    timesteps = 0

                self.memory.push(state, action, _state if _state is not None else end_state, reward)

                state = _state
                episode_loss = self.optimize()
                loss += episode_loss




            if episode % TARGET_UPDATE == 0:
                self.eval_network.update_params(self.network)
                print('episode ', episode, 'loss ', mean_loss, 'reward ', np.mean(sum_rewards))

            """
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr'] / (1 + (episode / LR_DECAY))
            """

if __name__ == '__main__':
    agent_smith = Agent()
    print('agent initiated')
    agent_smith.train(10000)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(timesalive)
    ax2.plot(losses)
    plt.show()
