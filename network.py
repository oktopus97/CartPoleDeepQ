from torch import nn
import torch.nn.functional as F
from torch import save, load


class DeepQNetwork(nn.Module):
    def __init__(self, no_inputs, no_outputs):

        # depth should be at least 0
        super(DeepQNetwork,self).__init__()
        self.no_inputs = no_inputs

        self.input_layer = nn.Linear(no_inputs,24)
        self.layer1 = nn.Linear(24,8)
        self.out_layer = nn.Linear(8,no_outputs)

    def forward(self, state):
        state = state.view(-1, self.no_inputs)
        state = F.relu(self.input_layer(state))
        state = F.relu(self.layer1(state))
        return F.softmax(self.out_layer(state))

    def getParameters(self):
        return tuple(self.parameters())

    def update_params(self, pred_net):
        state_dict = pred_net.state_dict()
        save(self.state_dict(),'params')
        self.load_state_dict(state_dict)

    def load_params(self, file='params'):
        print('loaded state dict')
        self.load_state_dict(load(file))
