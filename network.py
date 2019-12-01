from torch import nn
import torch.nn.functional as F
from torch import save, load
import random, string


class Network(nn.Module):
    def __init__(self, no_inputs, no_outputs):
        super(Network, self).__init__()
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs

    def getParameters(self):
        return tuple(self.parameters())

    def update_params(self, pred_net):
        state_dict = pred_net.state_dict()
        self.load_state_dict(state_dict)

    def load_params(self, file='params'):
        print('loaded state dict')
        self.load_state_dict(load(file))

    def save_params(self, agent_name, env_name):
        print('saved state dict')
        str = 'models/' + env_name + agent_name + \
        ''.join(random.choice(string.ascii_uppercase) for x in range(5))
        save(self.state_dict(), str)

class DeepQNetwork(Network):
    """
    returns a linear network
    NOT SUITED FOR LARGE INPUTS LIKE IMAGES
    """

    def __init__(self, no_inputs, no_outputs):
        super(DeepQNetwork,self).__init__(no_inputs, no_outputs)


        self.input_layer = nn.Linear(self.no_inputs, 24)
        self.layer1 = nn.Linear(24, 24)
        self.layer2 = nn.Linear(24, 8)
        self.out_layer = nn.Linear(8, self.no_outputs)

    def forward(self, state):
        state = state.view(-1, self.no_inputs)
        state = F.relu(self.input_layer(state))
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        return self.out_layer(state)
