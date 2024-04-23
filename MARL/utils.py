import torch
import torch as th
class Rollout:
    def __init__(self):
        self.node_id = [] # node id 
        self.state_lst = [] # state
        self.action_lst = [] # action
        self.logp_action_lst = [] # log proability of action
        self.reward_lst = [] # reward
        self.next_state_lst = [] # next state

    def put(self, state, action, logp_action, reward, done, next_state):
        self.state_lst.append(state)
        self.action_lst.append(action)
        self.logp_action_lst.append(logp_action)
        self.reward_lst.append(reward)
        self.done_lst.append(done)
        self.next_state_lst.append(next_state)
    
    # convert the stored data into tensor and return the tensor 
    def tensor(self):
        bs = torch.as_tensor(self.state_lst).float()
        ba = torch.as_tensor(self.action_lst).float()
        blogp_a = self.logp_action_lst
        br = torch.as_tensor(self.reward_lst).float()
        bd = torch.as_tensor(self.done_lst)
        bns = torch.as_tensor(self.next_state_lst).float()
        return bs, ba, blogp_a, br, bd, bns
    
def entropy(p):
    return -th.sum(p * th.log(p), 1)