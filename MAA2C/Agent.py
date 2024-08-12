from torch.distributions import Categorical
from ParameterConfig import MAA2C_Config
import torch
from MAA2C.Model import *
from MAA2C.utils import Rollout
class A2C:
    def __init__(self):
        self.V = ValueNet(MAA2C_Config.dim_global_observation) # value network
        self.V_target = ValueNet(MAA2C_Config.dim_global_observation) # target value network
        self.pi = PolicyNet(MAA2C_Config.dim_local_observation, MAA2C_Config.dim_action_sf) # policy network
        self.V_target.load_state_dict(self.V.state_dict()) # load the parameters of value network to target network
        self.value_optimizer = torch.optim.Adam(self.V.parameters(), lr=3e-4)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=3e-4)

        # load the networks to GPU
        self.V = self.V.to(MAA2C_Config.device)
        self.V_target = self.V_target.to(MAA2C_Config.device)
        self.pi = self.pi.to(MAA2C_Config.device)

        self.action = [0]
        self.logp_action = []
        self.partial_observation = []
        self.reward = 0
        self.cumulative_reward = 0
        self.next_partial_observation = []

        self.rollout_buffer = Rollout()   

    def get_action(self, observation):
        sf_log_probs = self.pi.policy(observation) # calculete the action proability of the given state with policy network
        
        # create categorical distribution object m
        m_sf = Categorical(sf_log_probs)
        
        # sample an action from m 
        action_sf = m_sf.sample()
        
        logp_action_sf = m_sf.log_prob(action_sf)

        action = [action_sf]
        return action, logp_action_sf

    def compute_value_loss(self, bgo, br, bngo):
        # target value predicted by the target value network with next state
        with torch.no_grad():
            target_value = br + MAA2C_Config.discount * self.V_target(bngo).squeeze()

        # calculate policy loss。
        with torch.no_grad():
            advantage = target_value - self.V(bgo).to(MAA2C_Config.device).squeeze()

        # calculate value loss
        value_loss = F.mse_loss(self.V(bgo).squeeze(), target_value)
        return value_loss, advantage

    def compute_policy_loss(self, blogp_a, advantage):
        policy_loss_sf = 0
        for i, logp_a in enumerate(blogp_a):
            policy_loss_sf += -logp_a * advantage[i]
        policy_loss_sf = policy_loss_sf.mean()
        return policy_loss_sf

    # soft update the parameters of target network to value network
    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.V_target, self.V, tau)
