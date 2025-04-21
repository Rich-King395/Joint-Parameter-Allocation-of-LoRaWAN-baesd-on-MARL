import ParameterConfig
from ParameterConfig import *
import numpy as np
class CMAB:
    def __init__(self):
        self.sf_arms = [0,1,2,3,4,5]

        self.sf_index = 0

        self.id = 0

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

class LinUCB(CMAB):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.d = 6    # 上下文维度，比如6
        self.alpha = alpha       # 探索系数

        # 初始化每个 arm 的 A 和 b
        self.A = [np.identity(self.d) for _ in range(self.K_SF)] # 协方差矩阵
        self.b = [np.zeros(self.d) for _ in range(self.K_SF)] # 累积奖励向量

        self.total_count = 0

    def actions_choose(self, context):
        self.total_count += 1

        p_vals = []
        for a in range(self.K_SF):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
            p_vals.append(p)
        
        # print("p_vals:", p_vals)

        k_sf = np.argmax(p_vals)
        self.counts_SF[k_sf] += 1

        self.action = [self.sf_arms[k_sf]]
        self.actions.append(self.action)

        action_choose = self.sf_arms[k_sf]
        return k_sf
    
    def update_linucb(self, context, reward, chosen_sf):
        self.A[chosen_sf] += np.outer(context, context)
        self.b[chosen_sf] += reward * context
        self.cumulative_reward_SF += reward



    


