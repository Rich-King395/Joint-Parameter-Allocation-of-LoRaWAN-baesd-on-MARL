import ParameterConfig
from ParameterConfig import *
import numpy as np
class MAB:
    def __init__(self):
        # each agent has three bandits
        self.K_SF = len(ParameterConfig.SF)
        self.K_BW = len(ParameterConfig.Bandwidth)
        self.K_Fre = len(ParameterConfig.Carrier_Frequency)
        self.K_TP = len(ParameterConfig.Transmission_Power)

        # intialize the expected reward of the handles of each bandit as 0
        self.Q_SF = np.zeros(self.K_SF, dtype=float)
        self.Q_BW = np.zeros(self.K_BW, dtype=float)
        self.Q_Fre = np.zeros(self.K_Fre, dtype=float)
        self.Q_TP = np.zeros(self.K_TP, dtype=float)
        
        # print(self.Q_SF,self.Q_BW,self.Q_Fre)
        # Number of the choices of each handdle of each bandit
        self.counts_SF = np.zeros(self.K_SF)
        self.counts_BW = np.zeros(self.K_BW)
        self.counts_Fre = np.zeros(self.K_Fre)
        self.counts_TP = np.zeros(self.K_TP)

        # reward for each step
        self.rewards = [0,0,0,0]
        
        # cumulative rewards of each LoRa resource of each agent
        self.cumulative_reward_SF = 0
        self.cumulative_reward_BW = 0
        self.cumulative_reward_Fre = 0
        self.cumulative_reward_TP = 0

        self.action = []
        # record the action chosen by the agent for each step
        self.actions = []

    def actions_choose(self):
        # the agent pull three handles for each step
        raise NotImplementedError
    
    def Expected_Reward_Update(self,k_sf,k_bw,k_fre,k_tp):
        # print("Update expected reward")
        # print("self.reward=",self.reward)
        # print("SF index",k_sf)

        self.rewards[0] = self.rewards[0] + float((SF[k_sf]/(2^SF[k_sf]))/SF_SUM)
        #print("self.rewards[0]=",float(self.rewards[0]))
        self.Q_SF[k_sf] += (self.rewards[0] - self.Q_SF[k_sf]) / (self.counts_SF[k_sf] + 1)
        
        self.rewards[1] = self.rewards[1] - float((Bandwidth[k_bw])/BW_SUM)
        self.Q_BW[k_bw] +=  (self.rewards[1] - self.Q_SF[k_bw]) / (self.counts_BW[k_bw] + 1)

        self.Q_Fre[k_fre] += (self.rewards[2] - self.Q_Fre[k_fre]) / (self.counts_Fre[k_fre] + 1)
    
        # self.rewards[3] = self.rewards[3] - float((ParameterConfig.Transmission_Power[k_tp]-8)/25)
        self.rewards[3] = self.rewards[3] - float(ParameterConfig.Transmission_Power[k_tp]/TP_SUM)*1.8
        self.Q_TP[k_tp] += (self.rewards[3] - self.Q_TP[k_tp]) / (self.counts_TP[k_tp] + 1)


""" epsilon greedy algorithm, inherit from MAB """
class EpsilonGreedy(MAB):
    def __init__(self):
        super(EpsilonGreedy, self).__init__()
        self.epsilon = MAB_Config.epsilon

    def actions_choose(self):
        '''SF choose'''
        if np.random.random() < self.epsilon:
            k_sf = np.random.randint(0, self.K_SF)  # randomly choose a handle
        else:
            k_sf = np.argmax(self.Q_SF)  # choose the handle with the highest reward
        self.counts_SF[k_sf] += 1
        '''Bandwidth choose'''
        if np.random.random() < self.epsilon:
            k_bw = np.random.randint(0, self.K_BW)  # randomly choose a handle
        else:
            k_bw = np.argmax(self.Q_BW)  # choose the handle with the highest reward
        self.counts_BW[k_bw] += 1
        '''Carrier frequency choose'''
        if np.random.random() < self.epsilon:
            k_fre = np.random.randint(0, self.K_Fre)  # randomly choose a handle
        else:
            k_fre = np.argmax(self.Q_Fre)  # choose the handle with the highest reward
        self.counts_Fre[k_fre] += 1
        '''Transmission power choose'''
        if np.random.random() < self.epsilon:
            k_tp = np.random.randint(0, self.K_TP)  # randomly choose a handle
        else:
            k_tp = np.argmax(self.Q_TP)  # choose the handle with the highest reward
        self.counts_TP[k_tp] += 1

        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre],ParameterConfig.Transmission_Power[k_tp]]
        '''store the actions for each step'''
        self.actions.append(self.action)

        return k_sf,k_bw,k_fre,k_tp

""" time-decreasing epsilon-greedy algorithm, inherit from MAB"""
class DecayingEpsilonGreedy(MAB):
    def __init__(self):
        super(DecayingEpsilonGreedy, self).__init__()
        self.epsilon = MAB_Config.decay_epsilon


    def actions_choose(self):
        '''SF choose'''
        if np.random.random() < self.epsilon:
            k_sf = np.random.randint(0, self.K_SF)  # randomly choose a handle
        else:
            k_sf = np.argmax(self.Q_SF)  # choose the handle with the highest reward
        self.counts_SF[k_sf] += 1
        '''Bandwidth choose'''
        if np.random.random() < self.epsilon:
            k_bw = np.random.randint(0, self.K_BW)  # randomly choose a handle
        else:
            k_bw = np.argmax(self.Q_BW)  # choose the handle with the highest reward
        self.counts_BW[k_bw] += 1
        '''Carrier frequency choose'''
        if np.random.random() < self.epsilon:
            k_fre = np.random.randint(0, self.K_Fre)  # randomly choose a handle
        else:
            k_fre = np.argmax(self.Q_Fre)  # choose the handle with the highest reward
        self.counts_Fre[k_fre] += 1
        '''Transmission power choose'''
        if np.random.random() < self.epsilon:
            k_tp = np.random.randint(0, self.K_TP)  # randomly choose a handle
        else:
            k_tp = np.argmax(self.Q_TP)  # choose the handle with the highest reward
        self.counts_TP[k_tp] += 1

        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre],ParameterConfig.Transmission_Power[k_tp]]
        '''store the actions for each step'''
        self.actions.append(self.action)

        return k_sf,k_bw,k_fre,k_tp


""" UCB(Upper Confidence Boundary) algorithm, inherit from MAB"""
class UCB(MAB):
    def __init__(self,coef):
        super(UCB, self).__init__()
        self.total_count = 0
        self.coef = coef

    def actions_choose(self):
        self.total_count += 1
        '''SF choose'''
        ucb_sf = self.Q_SF + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts_SF + 1)))  # calculate ucb of sf
        k_sf = np.argmax(ucb_sf)
        self.counts_SF[k_sf] += 1
        # print("SF",k_sf)
        # print(self.counts_SF[k_sf])
        '''Bandwidth choose'''
        ucb_bw = self.Q_BW + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts_BW + 1)))  # calculate ucb of bw
        k_bw = np.argmax(ucb_bw)
        self.counts_BW[k_bw] += 1
        '''Carrier frequency choose'''
        ucb_fre = self.Q_Fre + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts_Fre + 1)))  # calculate ucb of fre
        k_fre = np.argmax(ucb_fre)
        self.counts_Fre[k_fre] += 1
        '''Transmission power choose'''
        ucb_tp = self.Q_TP + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts_TP + 1)))  # calculate ucb of fre
        k_tp = np.argmax(ucb_tp)
        self.counts_TP[k_tp] += 1
       
        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre],ParameterConfig.Transmission_Power[k_tp]]
        '''store the actions for each step'''
        self.actions.append(self.action)

        return k_sf,k_bw,k_fre,k_tp