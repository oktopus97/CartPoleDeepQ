from hyper_params import MEMORY_CAPACITY,BATCH_SIZE

from collections import namedtuple
import random

from prep import prep_mem_batch


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    def __init__(self):
        self.capacity = MEMORY_CAPACITY
        self.memory = []

    def push(self,s,a,_s,r):
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

        self.memory.append(Transition(s, a, _s, r))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
