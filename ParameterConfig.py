"""
    LoRaWAN Parameters
"""
import numpy as np
import simpy
import matplotlib.pyplot as plt
import torch
import random
# turn on/off graphics
graphics = 0

# store the results or not
storage_flag = 1

# folder path of the results
result_folder_path = "/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL"

random_seed = 42

# do the full collision check
full_collision = False

# RSSI global values for antenna
dir_30 = 4
dir_90 = 2
dir_150 = -4
dir_180 = -3

# this is an array with measured values for sensitivity
# [SF,125KHz,250kHz,500kHz]
sf7 = np.array([7,-123,-120,-116])
sf8 = np.array([8,-126,-123,-119])
sf9 = np.array([9,-129,-125,-122])
sf10 = np.array([10,-132,-128,-125])
sf11 = np.array([11,-133,-130,-133])
sf12 = np.array([12,-136,-133,-130])

# receiver sensitivities of different SF and Bandwidth combinations
sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])

# maximum distance range of different settings (SF+Fre)
# [125KHz,250kHz,500kHz]
sf7_dis = np.array([1414,1243,1046])
sf8_dis = np.array([1610,1414,1190])
sf9_dis = np.array([1832,1542,1355])
sf10_dis = np.array([2085,1755,1542])
sf11_dis = np.array([2177,1913,1755])
sf12_dis = np.array([2477,2177,1913])

dis_range = np.array([sf7_dis,sf8_dis,sf9_dis,sf10_dis,sf11_dis,sf12_dis])

# minimum SNR required for demodulation at different spreading factors
SNR_Req = np.array([-7.5,-10,-12.5,-15,-17.5,-20])
Bandwidth = np.array([125,250,500])
SF = np.array([7,8,9,10,11,12])

Carrier_Frequency = np.array([470000,470100,470200,470300,470400,470500,470600,470700])
Transmission_Power = np.array([2,4,6,8,10,12,14])
SF_BW = [[7,125],[7,250],[7,500],
         [8,125],[8,250],[8,500],
         [9,125],[9,250],[9,500],
         [10,125],[10,250],[10,500],
         [11,125],[11,250],[11,500],
         [12,125],[12,250],[12,500]]

# adaptable LoRaWAN parameters to users
nrNodes = 200
nrBS = 1
radius = 1500
PayloadSize = 20
avgSendTime = 4000
allocation_type = "Local"
# allocation_method = "ADR"
allocation_method = "random"
# allocation_method = "Round Robin"
# allocation_method = "RS-LoRa"
# allocation_method = "MAB"
nrNetworks = 1
simtime = 1200000
directionality = 1
full_collision = True

traininterveltime = 400000

# global stuff
action_choose_interval = 30000 # configuration update time of MAA2C
interval_flag = 0
total_simtime = 0

nodes = [] # list of nodes
env = simpy.Environment() # simulation environment

# list of base stations
bs = []
# Packets sent to each GW
packetsAtBS = [] 
# Packets received by each GW
packetsRecBS = [] 
# list of received packets
recPackets=[]
# list of collided packets
collidedPackets=[]
# list of lost packets
lostPackets = []

RecPacketSize = 0
TotalPacketSize = 0
TotalPacketAirtime = 0
TotalEnergyConsumption = 0

# number of sent packets during update interval 
sentPackets_interval = 0
# number of received packets during update interval 
recPackets_interval = 0
# number of not received packets during update interval
lostPackets_interval = 0

PDRPerNode = [] # PDRs of all nodes in each episode
EnergyEfficiencyPerNode = [] # Energy efficiencies of all nodes in each episode

global_observation = []
next_global_observation = []

# global value of packet sequence numbers
packetSeq = 0

Ptx = 14 # packet transmission power
gamma = 2.32
d0 = 1000.0
std = 7.8           
Lpld0 = 128.95
GL = 0

# prepare graphics and add sink
# if (graphics == 1):
#     plt.ion()
#     plt.figure()
#     ax = plt.gcf().gca()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class LoRaParameters:
    sf = 12
    cr = 1
    bw = 125
    tp = 14
    fre = 470400
    PayloadSize = PayloadSize

class MAA2C_Config:
    num_agents = nrNodes
    dim_local_observation = 15 
    dim_global_observation = dim_local_observation * nrNodes
    dim_action_sf = SF.size
    dim_action_bw = Bandwidth.size
    dim_action_fre = Carrier_Frequency.size
    discount = 0.99 # discount coefficient
    num_episode = 2000
    receive_reward = 10
    lost_reward = -20
    fairness_weight = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed = 3

class MAB_Config:
    MAB_Variant = 2 # 0: epsilon-greedy, 1: decaying-greedy, 2: UCB
    
    coef = 2 # coefficient of UCB
    epsilon = 0.05
    decay_epsilon = 0.75
    random_seed = 2
    num_episode = 5000

    average_cumulative_reward = []

    average_cumulative_reward_SF = []
    average_cumulative_reward_BW = []
    average_cumulative_reward_Fre = []
    average_cumulative_reward_TP = []

    Network_PDR = [] # Network PDRs of all episodes
    Network_Throughput = [] # Network Throughputs of all episodes
    MinPDR = [] # Minimum PDR of all nodes in each episode
    MaxPDR = [] # Maximum PDR of all nodes in each episode
    MinEnergyEfficiency = [] # Minimum energy efficiency of all nodes in each episode
    MaxEnergyEfficiency = [] # Maximum energy efficiency of all nodes in each episode
    NetworkEnergyEfficiency = [] # Network energy efficiency of all nodes in each episode
    
    JainFairness = [] # Jain fairness of all nodes in each episode

class Q_table_Config:
    alpha = 0.1 # learning rate
    gamma = 0.9 # discount factor
    epsilon = 0.6 # initial greedy factor for decaying-greedy
    random_seed = 1
    buffer_size = 1000
    batch_size = 64
    num_episode = 4000
    experience_replay = False

def Jain_Fairness_Index(nodes):
    Suqare_of_Sum = 0
    Sum_of_Square = 0
    for node in nodes:
        Sum_of_Square += node.EnergyEfficiency ** 2
        Suqare_of_Sum += node.EnergyEfficiency
    Sum_of_Square = len(nodes) * (Sum_of_Square)
    Suqare_of_Sum = Suqare_of_Sum ** 2
    Jain_Fairness = float(Suqare_of_Sum) / float(Sum_of_Square)
    return Jain_Fairness

