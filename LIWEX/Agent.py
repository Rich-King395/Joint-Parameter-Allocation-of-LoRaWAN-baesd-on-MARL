import ParameterConfig
from ParameterConfig import *
import numpy as np
class LIWEX:
    def __init__(self):
        self.sf_arms = [0,1,2,3,4,5]
        self.tp_arms = Transmission_Power

        # number of actions
        self.K = len(self.sf_arms) * len(self.tp_arms)

        # intialize the expected reward of the base arms of each bandit as 0
        self.w_m = np.ones((len(self.sf_arms), len(self.tp_arms)))
        self.p_m = np.full((len(self.sf_arms), len(self.tp_arms)), 1 / self.K)
        
        self.T = 0

        self.gamma = 0
        # rewards for each step
        self.reward = 0
    
        self.action = []

        # record the action chosen by the agent for each step
        self.actions = []

    def actions_choose(self):
        self.T += 1

        rows, cols = self.p_m.shape
        K = self.p_m.size

        probabilities_flat = self.p_m.flatten()

        prob_sum = np.sum(probabilities_flat)
        if not np.isclose(prob_sum, 1.0):
            # print(f"Warning: Sum of probabilities is {prob_sum}. Normalizing.") # 可选的警告信息
            if prob_sum <= 0:
                 raise ValueError(f"Sum of probabilities is not positive ({prob_sum}). Cannot sample.")
            probabilities_flat = probabilities_flat / prob_sum
        
        chosen_index_1d = np.random.choice(np.arange(K), p=probabilities_flat)
        chosen_index_2d = np.unravel_index(chosen_index_1d, (rows, cols))

        sf_index = self.sf_arms[chosen_index_2d[0]]
        tp_index = chosen_index_2d[1]

        return sf_index, tp_index

    def Probability_Weight_Update(self, sf_index, tp_index):
        k_sf = self.sf_arms.index(sf_index)
        k_tp = tp_index
        self.gamma = min(1.0, math.sqrt((self.K * math.log2(self.K))/((math.e - 1) * self.T)))

        # Probaility Update
        U = self.w_m / np.sum(self.w_m)
        self.p_m = (1 - self.gamma) * U + self.gamma / self.K
        self.p_m /= np.sum(self.p_m)
        
        # Weight Update
        self.w_m[k_sf][k_tp] = self.w_m[k_sf][k_tp] * np.exp((self.gamma * self.reward) / (self.K * self.p_m[k_sf][k_tp]))

        # p_m_max = np.max(self.p_m)
        # for i in range(self.p_m.shape[0]):
        #     for j in range(self.p_m.shape[1]):
        #         if self.p_m[i, j] < 0.5 * p_m_max:
        #             self.p_m[i, j] = 0
        
        # self.p_m /= np.sum(self.p_m)


        





