from torch import nn
import torch.nn.functional as F
from torch import save, load

from hyper_params import HIDDEN_LAYER, NETWORK_DEPTH

class DeepQNetwork(nn.Module):
    def __init__(self, no_inputs, no_outputs, hidden_layer=HIDDEN_LAYER,depth=NETWORK_DEPTH):

        # depth should be at least 0
        super(DeepQNetwork,self).__init__()
        self.network = []
        self.no_inputs = no_inputs

        #Linear: y = Wx + b
        
        self.input_layer = nn.Linear(no_inputs,hidden_layer)


        for i in range(depth):

            self.network.append(nn.Linear(hidden_layer,hidden_layer))

        self.out_layer = nn.Linear(hidden_layer,no_outputs)

    def forward(self, state):
        state = state.view(-1, self.no_inputs)
        state = F.relu(self.input_layer(state))
        for layer in self.network:
            state = F.relu(layer(state))
        return F.softmax(self.out_layer(state))

    def getParameters(self):

        return tuple(self.parameters())

    def update_params(self, pred_net):
        state_dict = pred_net.state_dict()
        save(self.state_dict(),'params')
        self.load_state_dict(state_dict)

    def load_params(self):
        self.load_state_dict(load('params'))
        self.eval()