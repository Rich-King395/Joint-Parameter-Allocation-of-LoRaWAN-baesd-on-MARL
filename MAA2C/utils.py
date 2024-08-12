import torch
import torch as th
import numpy as np
import random
from ParameterConfig import  MAA2C_Config
class Rollout:
    def __init__(self):
        self.glocal_observation_lst = [] # glocal observation
        self.logp_action_lst = [] # log proability of action
        self.next_glocal_observation_lst = [] # next global observation
        self.reward_lst = [] # reward

    def put(self, glocal_observation, logp_action, next_glocal_observation, reward):
        self.glocal_observation_lst.append(glocal_observation)
        self.logp_action_lst.append(logp_action)
        self.next_glocal_observation_lst.append(next_glocal_observation)
        self.reward_lst.append(reward)
    
    # convert the stored data into tensor and return the tensor 
    def tensor(self):
        bgo = torch.as_tensor(self.glocal_observation_lst).float().to(MAA2C_Config.device)
        blogp_a = torch.as_tensor(self.logp_action_lst).float().to(MAA2C_Config.device)
        bngo = torch.as_tensor(self.next_glocal_observation_lst).float().to(MAA2C_Config.device)
        br = torch.as_tensor(self.reward_lst).float().to(MAA2C_Config.device)
        return bgo, blogp_a, bngo, br
    
def entropy(p):
    return -th.sum(p * th.log(p), 1)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True