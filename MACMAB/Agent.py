import ParameterConfig
from ParameterConfig import *
import numpy as np
class CMAB:
    def __init__(self):
        self.sf_arms = [0,1,2,3,4,5]

        # two base arms
        self.K_SF = len(self.sf_arms)

        # intialize the expected reward of the base arms of each bandit as 0
        self.Q_SF = np.zeros(self.K_SF, dtype=float)
        
        # Number of the choices of base arms of each bandit
        self.counts_SF = np.zeros(self.K_SF)

        # rewards for each step
        self.reward_SF = 0
        
        # cumulative rewards of each LoRa resource of each agent
        self.cumulative_reward_SF = 0

        self.action = []

        # record the action chosen by the agent for each step
        self.actions = []

    def actions_choose(self):
        # the agent pull three handles for each step
        raise NotImplementedError
    
    def Expected_Reward_Update(self, k_sf):        
        self.Q_SF[k_sf] += (self.reward_SF - self.Q_SF[k_sf]) / (self.counts_SF[k_sf] + 1)    


""" UCB(Upper Confidence Boundary) algorithm, inherit from CMAB"""
class CUCB(CMAB):
    def __init__(self):
        super(CUCB, self).__init__()
        self.total_count = 0
        
    def actions_choose(self):
        self.total_count += 1
        '''SF+BW choose'''
        ucb_sf = self.Q_SF + MACMAB_Config.maximum_sf_reward * np.sqrt(
            (3*np.log(self.total_count)) / (2 * (self.counts_SF + 1)))  # calculate ucb of sf
        k_sf = np.argmax(ucb_sf)
        self.counts_SF[k_sf] += 1
  
        '''store the actions for each step'''
        self.action = [SF[self.sf_arms[k_sf]]]
        self.actions.append(self.action)

        return k_sf



