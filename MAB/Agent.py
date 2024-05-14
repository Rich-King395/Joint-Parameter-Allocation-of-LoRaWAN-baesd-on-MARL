import ParameterConfig
import numpy as np
class MAB:
    def __init__(self):
        # each agent has three bandits
        self.K_SF = len(ParameterConfig.SF)
        self.K_BW = len(ParameterConfig.Bandwidth)
        self.K_Fre = len(ParameterConfig.Carrier_Frequency)
        # intialize the expected reward of the handles of each bandit as 0
        self.Q_SF = np.zeros(self.K_SF, dtype=float)
        self.Q_BW = np.zeros(self.K_BW, dtype=float)
        self.Q_Fre = np.zeros(self.K_Fre, dtype=float)
        
        # print(self.Q_SF,self.Q_BW,self.Q_Fre)
        # Number of the choices of each handdle of each bandit
        self.counts_SF = np.zeros(self.K_SF)
        self.counts_BW = np.zeros(self.K_BW)
        self.counts_Fre = np.zeros(self.K_Fre)

        # reward for each step
        self.reward = 0
        # cumulative reward for each agent
        self.cumulative_reward = 0

        self.action = []
        # record the action chosen by the agent for each step
        self.actions = []

    def actions_choose(self):
        # the agent pull three handles for each step
        raise NotImplementedError
    
    def Expected_Reward_Update(self,k_sf,k_bw,k_fre):
        # print("Update expected reward")
        # print("self.reward=",self.reward)
        # print("SF index",k_sf)
        self.Q_SF[k_sf] += (self.reward - self.Q_SF[k_sf]) / (self.counts_SF[k_sf] + 1)
        # print("self.reward - self.Q_SF[k_sf]=",self.reward - self.Q_SF[k_sf])
        # print("self.counts_SF[k_sf] + 1=",self.counts_SF[k_sf] + 1)
        # print("(self.reward - self.Q_SF[k_sf] / (self.counts_SF[k_sf] + 1)",((self.reward - self.Q_SF[k_sf]) / (self.counts_SF[k_sf] + 1)))
        # print("self.Q_SF[k_sf]",self.Q_SF[k_sf])
        self.Q_BW[k_bw] +=  (self.reward - self.Q_SF[k_bw]) / (self.counts_BW[k_bw] + 1)
        self.Q_Fre[k_fre] += (self.reward - self.Q_Fre[k_fre]) / (self.counts_Fre[k_fre] + 1)


""" epsilon greedy algorithm, inherit from MAB """
class EpsilonGreedy(MAB):
    def __init__(self, epsilon):
        super(EpsilonGreedy, self).__init__()
        self.epsilon = epsilon

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

        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre]]
        '''store the actions for each step'''
        self.actions.append(self.action)

        return k_sf,k_bw,k_fre

""" time-decreasing epsilon-greedy algorithm, inherit from MAB"""
class DecayingEpsilonGreedy(MAB):
    def __init__(self):
        super(DecayingEpsilonGreedy, self).__init__()
        self.total_count = 0

    def actions_choose(self):
        self.total_count += 1
        self.epsilon = 1 / self.total_count
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

        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre]]
        '''store the actions for each step'''
        self.actions.append(self.action)
        return k_sf,k_bw,k_fre


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

        self.action = [ParameterConfig.SF[k_sf],ParameterConfig.Bandwidth[k_bw],ParameterConfig.Carrier_Frequency[k_fre]]
        '''store the actions for each step'''
        self.actions.append(self.action)

        return k_sf,k_bw,k_fre