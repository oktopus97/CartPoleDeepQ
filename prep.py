"""
prep.py:

    file for pre-processing data.
"""
count= 0
from hyper_params import BATCH_SIZE, GAMMA
from torch.autograd import Variable

import torch
import torch.cuda
import numpy as np

from random import randrange
if torch.cuda.is_available():
    print('cuda')
    torch.cuda.init()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')


def prep_mem_batch(transitions):
    """
    returns gpu variables for input

    """


    state_batch, action_batch, _state_batch, reward_batch = zip(*transitions)
    state_batch    =  torch.tensor(state_batch).float().to(device)
    action_batch    = torch.tensor(action_batch).float().to(device)
    _state_batch   =  torch.tensor(_state_batch).float().to(device)
    reward_batch    = torch.tensor(reward_batch).float().to(device)
    return state_batch,action_batch,_state_batch,reward_batch


def prep_exploitation(state):
    state = torch.from_numpy(state).float().to(device)

    return state.unsqueeze(0)


def prep_exploration(action_space):
    return torch.tensor(randrange(int(action_space))).to(device)

def prep_q(max_next,reward):
    return max_next*GAMMA + reward
