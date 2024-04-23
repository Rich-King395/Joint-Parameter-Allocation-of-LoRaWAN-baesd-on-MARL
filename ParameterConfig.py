"""
    LoRaWAN Parameters
"""
import numpy as np
import simpy
import matplotlib.pyplot as plt
import torch
# turn on/off graphics
graphics = 1

# do the full collision check
full_collision = False

# RSSI global values for antenna
dir_30 = 4
dir_90 = 2
dir_150 = -4
dir_180 = -3

# this is an array with measured values for sensitivity
# [SF,125KHz,250kHz,500kHz]
sf7 = np.array([7,-126.5,-124.25,-120.75])
sf8 = np.array([8,-127.25,-126.75,-124.0])
sf9 = np.array([9,-131.25,-128.25,-127.5])
sf10 = np.array([10,-132.75,-130.25,-128.75])
sf11 = np.array([11,-134.5,-132.75,-128.75])
sf12 = np.array([12,-133.25,-132.25,-132.25])

# receiver sensitivities of different SF and Bandwidth combinations
sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])

# minimum SNR required for demodulation at different spreading factors
SNR_Req = np.array([-7.5,-10,-12.5,-15,-17.5,-20])

Bandwidth = np.array([125,250,500])
SF = np.array([7,8,9,10,11,12])
Carrier_Frequency = np.array([470000,470100,470200,470300,470400,470500,470600,470700])

# adaptable LoRaWAN parameters to users
nrNodes = 50
nrBS = 1
radius = 2000
PayloadSize = 65
avgSendTime = 2000
allocation_type = "Local"
allocation_method = "MARL"
nrNetworks = 1
simtime = 3600000
directionality = 1
full_collision = 1

# global stuff
update_interval = 100000 # configuration update time of MAA2C
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
if (graphics == 1):
    plt.ion()
    plt.figure()
    ax = plt.gcf().gca()


class LoRaParameters:
    sf = 9
    cr = 1
    bw = 125
    tp = 14
    fre = 868000000
    PayloadSize = PayloadSize

class MAA2C_Config:
    num_agents = nrNodes
    dim_local_observation = 12 
    dim_global_observation = dim_local_observation * nrNodes
    dim_action_sf = SF.size
    dim_action_bw = Bandwidth.size
    dim_action_fre = Carrier_Frequency.size
    discount = 0.99 # discount coefficient
    num_episode = 1500
    receive_reward = 10
    lost_reward = -5
    fairness_weight = 0.5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    