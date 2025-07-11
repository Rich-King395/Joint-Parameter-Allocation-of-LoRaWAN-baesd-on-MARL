"""
    LoRaWAN Parameters
"""
import numpy as np
import simpy
import matplotlib.pyplot as plt
import torch
import random
import math
import itertools

# turn on/off graphics
graphics = 1

# store the results or not
storage_flag = 1

# folder path of the results
result_folder_path = "/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL"

random_seed = 42

# do the full collision check
full_collision = True

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

# 按接收灵敏度从低到高排序的SF+BW组合
sf_bw_list = [
    (0, 2), (1, 2), (2, 2), (0, 1),
    (1, 1), (3, 2), (2, 1), (0, 0),
    (3, 1), (4, 2), (1, 0), (4, 1),
    (5, 2), (2, 0), (3, 0), (5, 1),
    (4, 0), (5, 0)
]
num_sf_bw = len(sf_bw_list)

ToA = np.array([97.536,174.592,328.704,616.448,1150.976,2138.112])
SF_SUM = float(2^7+2^8+2^9+2^10+2^11+2^12)
sf_sum = float(7/(2^7)+8/(2^8)+9/(2^9)+10/(2^10)+11/(2^11)+12/(2^12))
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
numSF = len(SF)

BW_SUM = 125 + 250 + 500
TP_SUM = 2 + 4 + 6 + 8 + 10 + 12 + 14

Carrier_Frequency = np.array([868100,868300,868500,868700,868900,869100,869300,869500])
numChannel = len(Carrier_Frequency)
Transmission_Power = np.array([2,4,6,8,10,12,14])
SF_BW = [[7,125],[7,250],[7,500],
         [8,125],[8,250],[8,500],
         [9,125],[9,250],[9,500],
         [10,125],[10,250],[10,500],
         [11,125],[11,250],[11,500],
         [12,125],[12,250],[12,500]]

All_Parameter_Configurations = itertools.product(Carrier_Frequency, SF, Transmission_Power)
Parameter_Configurations = list(All_Parameter_Configurations) # 336 configurations

sf_distribute = [0 for i in range(numSF)]
tp_distribute = [0 for i in range(len(Transmission_Power))]
fre_distribute = [0 for i in range(numChannel)]

# enable CASSI or not
CAASI_flag = 1

# 0 for homogeneous channel and 1 for heterogeneous channel
Channel_flag = 0

# adaptable LoRaWAN parameters to users
nrNodes = 200
nrBS = 1
radius = 1000
PayloadSize = 50
avgSendTime = 20000
allocation_type = "Local"
allocation_method = "ADR"
#allocation_method = "random"
#allocation_method = "uniform"
#allocation_method = "Round Robin"
#allocation_method = "RS-LoRa"
#allocation_method = "DLoRa"
#allocation_method = "MACMAB"
#allocation_method = "CDLoRa"
#allocation_method = "CoMAB"
#allocation_method = "MAConMAB"
#allocation_method = "LIWEX"
#allocation_method = "MIXMAB"
#allocation_method = "NaiveMAB"
#allocation_method = "EFLoRa"

global_episode = 0

nrNetworks = 1
simtime = 12000000
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
# nodes sent to each GW
packetsAtBS = [[] for _ in range(nrBS)]
# Packets' sequence number received by each GW
packetsRecBS = []
# list of sent packets 
sentPackets = [] 
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
ThroughputPerNode = [] # Throughputs of all nodes in each episode
MaxThroughput = 0 # Maximum throughput among the nodes (bits/s)
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
    fre = 868100
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

class DLoRa_Config:
    DLoRa_Variant = 2 # 0: epsilon-greedy, 1: decaying-greedy, 2: UCB
    
    coef = 2 # coefficient of UCB
    epsilon = 0.05
    decay_epsilon = 0.75
    random_seed = 2
    initialize_flag = 1

    initialize_duration = 12000000
    eposide_duration = 1800000 
    eval_duration = 12000000

    num_episode = 4000

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
    
    EEJainFairness = [] # Jain fairness of all nodes in each episode
    ThroughputJainFairness = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/DLoRa/results"

class Q_table_Config:
    alpha = 0.1 # learning rate
    gamma = 0.9 # discount factor
    epsilon = 0.6 # initial greedy factor for decaying-greedy
    random_seed = 1
    buffer_size = 1000
    batch_size = 64
    num_episode = 4000
    experience_replay = False

class CASSI_Config:
    n_period = math.ceil( nrNodes / numChannel )
    transmit_process = {}
    nodes_transmit = []

    packetsAtBS = [[] for _ in range(nrBS)]

    rssi_measurements = [[[] for _ in range(numChannel)] for _ in range(nrNodes)]
    Channel_RSSI = [] #用于储存按平均RSSI降序排序的信道列表
    Node_RSSI = [] #用于储存按平均RSSI升序排序的节点列表

    node_groups = [] # 排序分组后的节点，每组节点使用相同的信道

    sf_bw_PDR_thres = 25

    CF_time = 1200000
    SF_BW_time = 1200000

class MACMAB_Config:
    eposide_duration = 200000 
    tp_duration = 1200000
    eval_duration = 1200000
        
    maximum_sf_reward = 1

    node_rssi_list = [[] for _ in range(nrNodes)] #网关处用来储存所有节点数据包RSSI的列表

    sf_train_flag = 0

    sf_num_episode = 4000

    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MACMAB/results"

class CoCMAB_Config:
    eposide_duration = 200000 
    tp_duration = 1200000
    eval_duration = 1200000

    CoAgents = []
    joint_action_nodes = []
    joint_actions = []

    node_rssi_list = [[] for _ in range(nrNodes)] #网关处用来储存所有节点数据包RSSI的列表

    sf_train_flag = 0

    sf_num_episode = 2000
    tp_num_episode = 500 

    maximum_sf_bw_reward = 3           
    maximum_sf_reward = 2 
    maximum_tp_reward = 1.96

    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

class MAConMAB_Config:
    contexts = [np.array([0, 0, 0, 0, 0, 0]) for _ in range(numChannel)] # 初始化上下文向量
    eposide_duration = 2000000 
    tp_duration = 12000000
    eval_duration = 72000000

    node_rssi_list = [[] for _ in range(nrNodes)] #网关处用来储存所有节点数据包RSSI的列表

    sf_train_flag = 0

    sf_num_episode = 4000

    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MAConMAB/results"

class CDLoRa_Config:
    eposide_duration = 1800000 
    eval_duration = 12000000

    num_episode = 4000
        
    maximum_sf_reward = 2
    maximum_tp_reward = 2

    NetworkEnergyEfficiency = [] 
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/CDLoRa/results"

class NaiveMAB_Config:
    eposide_duration = 1800000 
    eval_duration = 12000000

    num_episode = 4000
        
    maximum_sf_reward = 2
    maximum_tp_reward = 2

    NetworkEnergyEfficiency = [] 
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/NaiveMAB/results"

class LIWEX_Config:
    Sum_ToA = np.sum(ToA)
    W_sf = Sum_ToA / ToA 
    W_sf *= 0.01

    N_TP = len(Transmission_Power)
    W_tp = np.zeros(N_TP) # Initialize an array for weights
    for j in range(1, N_TP + 1): # j runs from 1 to N_TP
        weight = math.ceil(1 - j / N_TP)
        # Store the weight. Note: array index is j-1 because Python uses 0-based indexing
        W_tp[j-1] = weight 

    W_m = W_sf[:, np.newaxis] * W_tp

    eposide_duration = 2000000 
    eval_duration = 12000000

    num_episode = 2000
        
    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/LIWEX/results"

class MIXMAB_Config:
    eposide_duration = 1800000  
    eval_duration = 12000000

    num_episode = 4000
        
    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MIXMAB/results"

class EFLoRa_Config:
    eposide_duration = 1800000  
    eval_duration = 12000000

    num_episode = 0

    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/EFLoRa/results"

class ADR_Config:
    Nodes_RSSI = [[] for _ in range(nrNodes)]

    eposide_duration = 1800000  
    eval_duration = 12000000

    num_episode = 1000

    NetworkEnergyEfficiency = []
    Network_PDR = []
    Network_Throughput = []

    NetPDR = 0
    NetThroughput = 0
    NetEnergyEfficiency = 0

    result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/ADR/results"


def EE_Jain_Fairness_Index(nodes):
    Suqare_of_Sum = 0
    Sum_of_Square = 0
    for node in nodes:
        Sum_of_Square += node.EnergyEfficiency ** 2
        Suqare_of_Sum += node.EnergyEfficiency
    Sum_of_Square = len(nodes) * (Sum_of_Square)
    Suqare_of_Sum = Suqare_of_Sum ** 2
    EE_Jain_Fairness = float(Suqare_of_Sum) / float(Sum_of_Square)
    return EE_Jain_Fairness

def Throughput_Jain_Fairness_Index(nodes):
    Suqare_of_Sum = 0
    Sum_of_Square = 0
    for node in nodes:
        Sum_of_Square += node.Throughput ** 2
        Suqare_of_Sum += node.Throughput
    Sum_of_Square = len(nodes) * (Sum_of_Square)
    Suqare_of_Sum = Suqare_of_Sum ** 2
    Throughput_Jain_Fairness = float(Suqare_of_Sum) / float(Sum_of_Square)
    return Throughput_Jain_Fairness

def Linear_Product_based_Fairness_Index(nodes):
    L = 1
    for node in nodes:
        L = L * float(node.Throughput/MaxThroughput)
    return L

def Throughput_Variance(nodes):
    sum = 0
    for node in nodes:
        sum += node.Throughput
    mean = sum / len(nodes)
    variance = 0
    for node in nodes:
        variance += (node.Throughput - mean) ** 2
    variance = variance / len(nodes)
    return variance
