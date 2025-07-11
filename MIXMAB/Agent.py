import ParameterConfig
from ParameterConfig import *
import numpy as np
class MIXMAB:
    def __init__(self):
        self.arms = Parameter_Configurations

        # number of actions
        self.K = len(self.arms)

        # intialize the expected reward of the base arms of each bandit as 0
        self.N = np.zeros(self.K, dtype=float)
        self.P = np.zeros(self.K, dtype=float)
        self.W = np.ones(self.K, dtype=float)
               
        self.l_EXP = 5
        self.alpha = 1
        self.l_EE = 100

        self.round_robin_index = 0

        self.T = 0
        # rewards for each step
        self.reward = 0

    def actions_choose(self):
        self.T += 1
        self.gamma = min(1.0, math.sqrt((self.K * math.log2(self.K))/((math.e - 1) * self.T)))

        if self.N.min() <= self.l_EXP:
            arm_index = self.round_robin_index
            self.round_robin_index += 1
            if self.round_robin_index > 335:
                self.round_robin_index = 0
        else:
            arm_index = np.random.choice(np.arange(self.K), p=self.P)
        return arm_index

    def Probability_Weight_Update(self, arm_index):
        # Probaility Update
        U = self.W[arm_index] / np.sum(self.W)
        self.P[arm_index] = (1 - self.gamma) * U + self.gamma / self.K
        
        self.P /= np.sum(self.P)
        
        # Weight Update
        self.W[arm_index] = self.W[arm_index] * np.exp((self.gamma * self.reward) / (self.K * self.P[arm_index]))

        self.N[arm_index] += 1 
        if self.N[arm_index] > self.l_EXP:
            if self.P[arm_index] < 0.5*self.P.max():
                self.P[arm_index] = 0
                self.P /= np.sum(self.P)
        if self.N[arm_index] > self.alpha*self.l_EE:
            self.N = np.zeros(self.K, dtype=float)
            self.alpha += 1
        



    


        





