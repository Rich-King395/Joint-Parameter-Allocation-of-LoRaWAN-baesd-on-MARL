import ParameterConfig
from ParameterConfig import *
import numpy as np
class NaiveMAB:
    def __init__(self):
        self.arms = Parameter_Configurations

        # number of actions
        self.K = len(self.arms)

        # intialize the expected reward of the base arms of each bandit as 0
        self.T = np.zeros(self.K, dtype=float)
        self.Q = np.zeros(self.K, dtype=float)

        self.l_EXP = 5
        
        self.round_robin_index = 0

        self.t = 0
        # rewards for each step
        self.reward = 0

        self.explore_flag = True

    def actions_choose(self):
        if self.T.min() <= self.l_EXP:
            arm_index = self.round_robin_index
            self.round_robin_index += 1
            if self.round_robin_index > 335:
                self.round_robin_index = 0
        else:
            # print("开始学习：")
            self.t += 1
            
            if self.explore_flag == True:
                self.T = np.zeros(self.K, dtype=float)
                self.explore_flag = False
            ucb = self.Q + 2 * np.sqrt(np.log(self.t) / (2 * (self.T + 1)))
            arm_index = np.argmax(ucb)
        return arm_index

    def Expected_Reward_Update(self, arm_index):
        self.T[arm_index] += 1 
        self.Q[arm_index] += (self.reward - self.Q[arm_index]) / (self.T[arm_index] + 1)
        
        



    


        





