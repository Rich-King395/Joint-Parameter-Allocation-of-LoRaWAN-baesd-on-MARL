import torch
import torch.nn as nn
import torch.nn.functional as F
from ParameterConfig import MAA2C_Config
'''
    value network
    3 full-connection layers
    relu activation function
'''
class ValueNet(nn.Module):
    def __init__(self, dim_state):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 258)
        self.fc2 = nn.Linear(258, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
    Policy network
    3 full-connection network
    relu activation function
    output the proability distribution of each action 
'''
class PolicyNet(nn.Module):
    def __init__(self, dim_state, num_action_sf):
        super().__init__()
        # shared two full connection layers
        self.fc1=nn.Linear(dim_state, 128)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64, 32)
        self.fc4=nn.Linear(32, num_action_sf)
        
    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out_sf=self.fc4(x)

        sf_prob = F.softmax(out_sf, dim=-1)
        return sf_prob
    
    def policy(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out_sf=self.fc4(x)
        
        sf_log_prob = F.log_softmax(out_sf, dim=-1)
        return sf_log_prob



