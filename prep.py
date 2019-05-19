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
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
##is_available is not working with my current hardware :(

def prep_mem_batch(transitions):
    """
    returns gpu variables for input

    """

    state_batch, action_batch, _state_batch, reward_batch = zip(*transitions)
    state_batch    =  torch.tensor(state_batch).float()
    action_batch    = torch.tensor(action_batch).float()
    _state_batch   =  torch.tensor(_state_batch).float()
    reward_batch    = torch.tensor(reward_batch).float()
    return state_batch,action_batch,_state_batch,reward_batch


def prep_exploitation(state):
    state = torch.from_numpy(state).float()

    return state.unsqueeze(0)


def prep_exploration(action_space):
    return torch.tensor(randrange(int(action_space)))

def prep_q(max_next,reward_batch):
    q =  max_next*GAMMA + reward_batch
    return q