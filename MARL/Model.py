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
    def __init__(self, dim_state, num_action_bw, num_action_sf, num_action_fre):
        super().__init__()
        # shared two full connection layers
        self.shared_layers = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # output layer of bw
        self.output_bw = nn.Linear(32, num_action_bw)
        
        # output layer of sf
        self.output_sf = nn.Linear(32, num_action_sf)
        
        # output layer of frequency
        self.output_fre = nn.Linear(32, num_action_fre)

    def forward(self, observation):
        x = self.shared_layers(observation)
        out_bw = self.output_bw(x)
        out_sf = self.output_sf(x)
        out_fre = self.output_fre(x)
        
        bw_prob = F.softmax(out_bw, dim=-1)
        sf_prob = F.softmax(out_sf, dim=-1)
        fre_prob = F.softmax(out_fre, dim=-1)
        
        return bw_prob, sf_prob, fre_prob
    
    def policy(self, observation):
        x = self.shared_layers(observation)
        out_bw = self.output_bw(x)
        out_sf = self.output_sf(x)
        out_fre = self.output_fre(x)
        
        bw_log_prob = F.log_softmax(out_bw, dim=-1)
        sf_log_prob = F.log_softmax(out_sf, dim=-1)
        fre_log_prob = F.log_softmax(out_fre, dim=-1)
        
        return bw_log_prob, sf_log_prob, fre_log_prob



