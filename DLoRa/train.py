import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
from Node import *
from Gateway import *
from plot import *
import random

global first_record_call 
first_record_call = True
def DLoRa_run(nodes):
    # set_seed(random_seed)

    # DLoRa_initialize(nodes)
    DLoRa_Config.initialize_flag = 0

    print("初始化阶段结束开始训练...")
    for episode in range(DLoRa_Config.num_episode):
        ParameterConfig.global_episode = episode
        DLoRa_train(nodes, episode)
        

    training_process(DLoRa_Config)

    DLoRa_eval(nodes)

    result_record(DLoRa_Config.NetPDR, DLoRa_Config.NetEnergyEfficiency)  

def DLoRa_initialize(nodes):
    env = simpy.Environment()

    for node in nodes:
        node.sent = 0
        node.packetloss = 0
        node.packetrec = 0

        node.PDR = 0 # packet delivery ratio of the node
        node.RecPacketSize = 0 # size of packets received by the node
        node.EnergyConsumption = 0 # energy consumption of the node
        node.EnergyEfficiency = 0 # energy efficiency of the node
        node.TotalPacketAirtime = 0 # total packet airtime of the node
        node.Throughput = 0 # throughput of the node

        env.process(transmit(env,node))

    DLoRa_Config.initialize_flag = 0
    env.run(until=DLoRa_Config.initialize_duration)

def DLoRa_train(nodes, episode):

    # initialize simulation environment current time for each episode
    env = simpy.Environment()

    ''' initialize the environment at the beginning of episode '''
    reset_simulation_stats()

    for node in nodes:
        node.sent = 0
        node.packetloss = 0
        node.packetrec = 0

        node.PDR = 0 # packet delivery ratio of the node
        node.RecPacketSize = 0 # size of packets received by the node
        node.EnergyConsumption = 0 # energy consumption of the node
        node.EnergyEfficiency = 0 # energy efficiency of the node
        node.TotalPacketAirtime = 0 # total packet airtime of the node
        node.Throughput = 0 # throughput of the node

        env.process(transmit(env,node))
    
    env.run(until=DLoRa_Config.eposide_duration)

    if DLoRa_Config.DLoRa_Variant == 1:
        DLoRa_Config.decay_epsilon -=  float(DLoRa_Config.decay_epsilon / DLoRa_Config.num_episode)


    sumRec = 0
    sumSent = 0
    for node in nodes:
        node.PDR = float((node.packetrec)/(node.sent))
        ParameterConfig.PDRPerNode.append(node.PDR)
        node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
        ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
        node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
        ParameterConfig.ThroughputPerNode.append(node.Throughput)

        sumRec += node.packetrec
        sumSent += node.sent

    NetPDR = float(sumRec/sumSent) 
    NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    DLoRa_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    DLoRa_Config.Network_PDR.append(NetPDR)
    DLoRa_Config.Network_Throughput.append(NetThroughput)

    # print(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, Network Throughput={NetThroughput:.2f}, Network EE={NetEnergyEfficiency:.2f}, PDR={pdr:.2f}, Throughput Variance={ThroughputVariance:.3f}")
    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")
    record(episode, NetPDR, NetEnergyEfficiency)

'''训练结束后, 测试训练效果'''
def DLoRa_eval(nodes):
    # initialize simulation environment current time for each episode
    env = simpy.Environment()

    ''' initialize the environment at the beginning of episode '''
    reset_simulation_stats()

    ''' 所有节点在传输进程开始前数据初始化并进行动作选择 '''
    for node in nodes:
        node.sent = 0
        node.packetloss = 0
        node.packetrec = 0

        node.PDR = 0 # packet delivery ratio of the node
        node.RecPacketSize = 0 # size of packets received by the node
        node.EnergyConsumption = 0 # energy consumption of the node
        node.EnergyEfficiency = 0 # energy efficiency of the node
        node.TotalPacketAirtime = 0 # total packet airtime of the node
        node.Throughput = 0 # throughput of the node

        ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
        env.process(transmit(env,node))

    env.run(until=DLoRa_Config.eval_duration)


    for node in nodes:
        # print("node.id:",node.id,"SF:",SF[node.sf_index])
        sf_distribute[node.sf_index] += 1
        tp_distribute[node.tp_index] += 1
        fre_distribute[node.fre_index] += 1

    DLoRa_Config.NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    DLoRa_Config.NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    DLoRa_Config.NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    sf_distribution(DLoRa_Config.result_folder_path)
    tp_distribution(DLoRa_Config.result_folder_path)

    print("Evaluation结束")
    print(f"Evaluation得到的网络指标为: PDR={DLoRa_Config.NetPDR*100:.2f}, Network EE={DLoRa_Config.NetEnergyEfficiency:.2f}, Network Throughput={DLoRa_Config.NetThroughput:.2f},")

def reset_simulation_stats():
    """重置仿真统计量，确保每次动作测试独立"""
    # Packets sent to each GW
    packetsAtBS = [[] for _ in range(nrBS)]
    ParameterConfig.packetsRecBS = []
    ParameterConfig.sentPackets = []
    ParameterConfig.recPackets = []
    ParameterConfig.collidedPackets = []
    ParameterConfig.lostPackets = []

    ParameterConfig.RecPacketSize = 0
    ParameterConfig.TotalPacketSize = 0
    ParameterConfig.TotalPacketAirtime = 0
    ParameterConfig.TotalEnergyConsumption = 0


    ParameterConfig.PDRPerNode = []
    ParameterConfig.EnergyEfficiencyPerNode = []
    ParameterConfig.ThroughputPerNode = []

def record(episode, PDR, EE):
    base_result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/DLoRa/results"
    
    if CAASI_flag == 1:
        subfolder_name = f"{nrNodes}_nodes_{radius}_m_CAASI"
        if Channel_flag == 0:
            subfolder_name = f"{nrNodes}_nodes_{radius}_m_CAASI_HomoChannel"
        else:
            subfolder_name = f"{nrNodes}_nodes_{radius}_m_CAASI_HeterChannel"
    else:
        subfolder_name = f"{nrNodes}_nodes_{radius}_m"
        if Channel_flag == 0:
            subfolder_name = f"{nrNodes}_nodes_{radius}_m_HomoChannel"
        else:
            subfolder_name = f"{nrNodes}_nodes_{radius}_m_HeterChannel"
    

    DLoRa_Config.result_folder_path = os.path.join(base_result_folder_path, subfolder_name)

    if not os.path.exists(DLoRa_Config.result_folder_path):
        os.makedirs(DLoRa_Config.result_folder_path)
    
    record_file = "record.txt"
    record_file_path = os.path.join(DLoRa_Config.result_folder_path, record_file)

    global first_record_call
    mode = 'w' if first_record_call else 'a'
    with open(record_file_path, mode) as file:
        file.write(f"episode={episode} PDR={PDR*100:.2f} EE={EE:.2f}\n")
    
    first_record_call = False  

def result_record(PDR, EE):
    result_file = "Result.txt"
    result_file_path = os.path.join(DLoRa_Config.result_folder_path, result_file)
    with open(result_file_path, 'w') as file:
        file.write('--------Parameter Setting--------\n')
        file.write('Nodes per base station: {}\n'.format(nrNodes))
        file.write('Packet generation interval: {} ms\n'.format(avgSendTime))
        file.write('LoRa parameters allocation type: {}\n'.format(allocation_type))
        file.write('LoRa parameters allocation method: {}\n'.format(allocation_method))
        file.write('Simulation duration: {} h\n'.format(int(simtime/3600000)))
        file.write('Number of gateways: {}\n'.format(nrBS))
        if full_collision == 1:
            file.write('Collision check mode: Full Collision Check\n')
        else:
            file.write('Collision check mode: Simple Collision Check\n')
        if directionality == 1:
            file.write('Antenna type: Directional antenna\n')
        else:
            file.write('Antenna type: Omnidirectional antenna\n')
        file.write('Number of networks: {}\n'.format(nrNetworks))
        file.write('Network topology radius: {} m\n'.format(radius))
        file.write('Packet payload size: {}\n\n'.format(PayloadSize))

        file.write('--------Simulation Results--------\n')
        file.write('Results after power control:\n')
        file.write('PDR={:.3f} %,'.format(PDR*100))
        file.write('Network EE={:.3f} bits/mJ\n'.format(EE))
        file.write('--------Parameter Distribution--------\n')
        for i, count in enumerate(sf_distribute):
            file.write('SF={} -> {}\n'.format(7 + i, count))  # SF starts from 7
        for i, count in enumerate(fre_distribute):
            file.write('Freq={} kHz -> {}\n'.format(Carrier_Frequency[i], count))
        for i, count in enumerate(tp_distribute):
            file.write('TP={} dBm -> {}\n'.format(Transmission_Power[i], count))
    

        


