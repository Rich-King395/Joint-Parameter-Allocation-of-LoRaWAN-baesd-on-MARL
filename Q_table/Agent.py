import ParameterConfig
from ParameterConfig import Q_table_Config
import numpy as np
import random
class Q_table:
    def __init__(self):
        # each agent has three bandits
        self.K_SF = len(ParameterConfig.SF)
        self.K_BW = len(ParameterConfig.Bandwidth)
        self.K_Fre = len(ParameterConfig.Carrier_Frequency)
        # three Q-table of the agent
        self.Q_table_SF = np.zeros(self.K_SF, dtype=float)
        self.Q_table_BW = np.zeros(self.K_BW, dtype=float)
        self.Q_table_Fre = np.zeros(self.K_Fre, dtype=float)
        self.alpha = Q_table_Config.alpha  # learning rate
        self.gamma = Q_table_Config.gamma  # discount factor
        self.epsilon = Q_table_Config.epsilon  # greedy factor
        # replay buffer of the agent
        self.buffer = RelpayBuffer(Q_table_Config.buffer_size)

        self.reward = 0
        # cumulative reward for each agent
        self.cumulative_reward = 0

    def actions_choose(self):
        if self.epsilon > 0.01:
            self.epsilon -= 0.005 # decaying greedy
        else:
            pass
        '''SF choose'''
        if np.random.random() < self.epsilon:
            k_sf = np.random.randint(0, self.K_SF)
        else:
            k_sf = np.argmax(self.Q_table_SF)
        '''Bandwidth choose'''
        if np.random.random() < self.epsilon:
            k_bw = np.random.randint(0, self.K_BW)
        else:
            k_bw = np.argmax(self.Q_table_BW)
        '''Carrier frequency choose'''
        if np.random.random() < self.epsilon:
            k_sf = np.random.randint(0, self.K_SF)
        else:
            k_sf = np.argmax(self.Q_table_SF)
        return k_sf,k_bw,k_sf
    
    def update_with_experience_replay(self):
        if len(self.buffer.memory)<Q_table_Config.batch_size:
            return
        batch = self.buffer.sample(Q_table_Config.batch_size)
        for experience in batch:
            '''update SF table'''
            TD_error_SF = experience[1] +self.gamma*self.Q_table_SF.max()-self.Q_table_SF[experience[0][0]]
            self.Q_table_SF[experience[0][0]] += self.alpha * TD_error_SF
            '''update BW table'''
            TD_error_BW = experience[1] +self.gamma*self.Q_table_BW.max()-self.Q_table_BW[experience[0][1]]
            self.Q_table_BW[experience[0][1]] += self.alpha * TD_error_BW
            '''update SF table'''
            TD_error_Fre = experience[1] +self.gamma*self.Q_table_Fre.max()-self.Q_table_Fre[experience[0][2]]
            self.Q_table_Fre[experience[0][2]] += self.alpha * TD_error_Fre
    
    def update_without_experience_replay(self,actions):
        '''update SF table'''
        TD_error_SF = self.reward +self.gamma*self.Q_table_SF.max()-self.Q_table_SF[actions[0]]
        self.Q_table_SF[actions[0]] += self.alpha * TD_error_SF
        '''update BW table'''
        TD_error_BW = self.reward +self.gamma*self.Q_table_BW.max()-self.Q_table_BW[actions[1]]
        self.Q_table_BW[actions[1]] += self.alpha * TD_error_BW
        '''update SF table'''
        TD_error_Fre = self.reward +self.gamma*self.Q_table_Fre.max()-self.Q_table_Fre[actions[2]]
        self.Q_table_Fre[actions[2]] += self.alpha * TD_error_Fre
    
class RelpayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, actions, reward):
        """Saves a action and reward"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [actions, reward]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
