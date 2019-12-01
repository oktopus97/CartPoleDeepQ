"""
gives all the necessary hyper parameters

"""
LR = .001
LR_DECAY = 5000
GAMMA = .95              # reward discount
MEMORY_CAPACITY = 5000


BATCH_SIZE = 64
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = .99
TARGET_UPDATE = 100          #eval_net update interval in cycles
