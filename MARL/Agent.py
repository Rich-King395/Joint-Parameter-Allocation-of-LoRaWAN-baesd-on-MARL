from torch.distributions import Categorical
from ParameterConfig import MAA2C_Config
import torch
from MARL.Model import *

class A2C:
    def __init__(self):
        self.V = ValueNet(MAA2C_Config.dim_global_observation) # value network
        self.V_target = ValueNet(MAA2C_Config.dim_global_observation) # target value network
        self.pi = PolicyNet(MAA2C_Config.dim_local_observation, MAA2C_Config.dim_action_bw, MAA2C_Config.dim_action_sf, MAA2C_Config.dim_action_fre) # policy network
        self.V_target.load_state_dict(self.V.state_dict()) # load the parameters of value network to target network
        self.value_optimizer = torch.optim.Adam(self.V.parameters(), lr=3e-3)
        self.pi_optimizer = torch.optim.Adam(self.pi.shared_layers.parameters(), lr=3e-3)
        self.pi_optimizer_output_bw = torch.optim.Adam(self.pi.output_bw.parameters(), lr=3e-3)
        self.pi_optimizer_output_sf = torch.optim.Adam(self.pi.output_sf.parameters(), lr=3e-3)
        self.pi_optimizer_output_fre = torch.optim.Adam(self.pi.output_fre.parameters(), lr=3e-3)

        # load the networks to GPU
        self.V = self.V.to(MAA2C_Config.device)
        self.V_target = self.V_target.to(MAA2C_Config.device)
        self.pi = self.pi.to(MAA2C_Config.device)

        self.action = [0,0,0]
        self.logp_action = []
        self.partial_observation = []
        self.reward = 0
        self.next_partial_observation = []



    def get_action(self, observation):
        bw_log_probs, sf_log_probs, fre_log_probs = self.pi.policy(observation) # calculete the action proability of the given state with policy network
        m_bw = Categorical(bw_log_probs) # create categorical distribution object m
        m_sf = Categorical(sf_log_probs)
        m_fre = Categorical(fre_log_probs)
        action_bw = m_bw.sample() # sample an action from m 
        action_sf = m_sf.sample()
        action_fre = m_fre.sample()
        logp_action_bw = m_bw.log_prob(action_bw)
        logp_action_sf = m_sf.log_prob(action_sf)
        logp_action_fre = m_fre.log_prob(action_fre)
        action = [action_bw, action_sf, action_fre]
        logp_action = [logp_action_bw, logp_action_sf, logp_action_fre]
        return action, logp_action

    def compute_value_loss(self, bs, br, bns):
        # target value predicted by the target value network with next state
        with torch.no_grad():
            target_value = br + MAA2C_Config.discount * self.V_target(bns).squeeze()

        # calculate policy loss。
        with torch.no_grad():
            advantage = target_value - self.V(bs).squeeze()

        # calculate value loss
        value_loss = F.mse_loss(self.V(bs).squeeze(), target_value)
        return value_loss, advantage

    def compute_policy_loss(self, logp_a, advantage):
        policy_loss_bw = 0
        policy_loss_sf = 0
        policy_loss_fre = 0
        policy_loss_bw += -logp_a[0] * advantage
        policy_loss_sf += -logp_a[1] * advantage
        policy_loss_fre += -logp_a[2] * advantage
        return policy_loss_bw, policy_loss_sf, policy_loss_fre

    # soft update the parameters of target network to value network
    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.V_target, self.V, tau)
