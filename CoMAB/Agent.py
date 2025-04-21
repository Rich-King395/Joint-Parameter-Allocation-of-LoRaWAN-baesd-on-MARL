import ParameterConfig
from ParameterConfig import *
import numpy as np
class CMAB:
    def __init__(self):
        self.sf_arms = [0,1,2,3,4,5]
        self.tp_arms = Transmission_Power

        # two base arms
        self.K_TP = len(self.tp_arms)

        # intialize the expected reward of the base arms of each bandit as 0
        self.Q_TP = np.zeros(self.K_TP, dtype=float)
        
        # Number of the choices of base arms of each bandit
        self.counts_TP = np.zeros(self.K_TP)

        # rewards for each step
        self.reward_TP = 0
        
        # cumulative rewards of each LoRa resource of each agent
        self.cumulative_reward_TP = 0

        self.action = []

        # record the action chosen by the agent for each step
        self.actions = []

    def actions_choose(self):
        # the agent pull three handles for each step
        raise NotImplementedError
    
    def Expected_Reward_Update(self, k_tp):          
        self.Q_TP[k_tp] += (self.reward_TP - self.Q_TP[k_tp]) / (self.counts_TP[k_tp] + 1)


""" UCB(Upper Confidence Boundary) algorithm, inherit from CMAB"""
class CoCUCB(CMAB):
    def __init__(self):
        super(CoCUCB, self).__init__()
        self.total_count = 0
        
    def actions_choose(self):
        self.total_count += 1

        '''Transmission power choose'''
        ucb_tp = self.Q_TP + MACMAB_Config.maximum_tp_reward * np.sqrt(
            (3*np.log(self.total_count)) / (2 * (self.counts_TP + 1)))  # calculate ucb of fre
        k_tp = np.argmax(ucb_tp)
        self.counts_TP[k_tp] += 1
       
        '''store the actions for each step'''
        self.action = [Transmission_Power[k_tp]]
        self.actions.append(self.action)

        return k_tp



