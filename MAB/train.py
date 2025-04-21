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
import random
def MAB_run(nodes):
    # set_seed(random_seed)

    MAB_initialize(nodes)
    print("初始化阶段结束开始训练...")
    for episode in range(MAB_Config.num_episode):
        MAB_train(nodes, episode)

    plot()

    MAB_eval(nodes)

    result_record(MAB_Config.NetPDR, MAB_Config.NetEnergyEfficiency)  

def MAB_initialize(nodes):
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

    MAB_Config.initialize_flag = 0
    env.run(until=MAB_Config.initialize_duration)

def MAB_train(nodes, episode):

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
    
    env.run(until=MAB_Config.eposide_duration)

    if MAB_Config.MAB_Variant == 1:
        MAB_Config.decay_epsilon -=  float(MAB_Config.decay_epsilon / MAB_Config.num_episode)


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

    MAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MAB_Config.Network_PDR.append(NetPDR)
    MAB_Config.Network_Throughput.append(NetThroughput)

    # print(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, Network Throughput={NetThroughput:.2f}, Network EE={NetEnergyEfficiency:.2f}, PDR={pdr:.2f}, Throughput Variance={ThroughputVariance:.3f}")
    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")
        
'''训练结束后, 测试训练效果'''
def MAB_eval(nodes):
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

    env.run(until=MAB_Config.eval_duration)

    for node in nodes:
        # print("node.id:",node.id,"SF:",SF[node.sf_index])
        sf_distribute[node.sf_index] += 1
        tp_distribute[node.tp_index] += 1
        fre_distribute[node.fre_index] += 1

    MAB_Config.NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    MAB_Config.NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    MAB_Config.NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    print("Evaluation结束")
    print(f"Evaluation得到的网络指标为: PDR={MAB_Config.NetPDR*100:.2f}, Network EE={MAB_Config.NetEnergyEfficiency:.2f}, Network Throughput={MAB_Config.NetThroughput:.2f},")

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


def plot():
    base_result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MAB/results"
    subfolder_name = f"{nrNodes}_nodes_{radius}_m"
    MAB_Config.result_folder_path = os.path.join(base_result_folder_path, subfolder_name)

    # 如果文件夹不存在则创建
    if not os.path.exists(MAB_Config.result_folder_path):
        os.makedirs(MAB_Config.result_folder_path)

    '''Network Energy Efficiency'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.NetworkEnergyEfficiency, label='NetworkEnergyEfficiency', color='blue')

    plt.title('Network Energy Efficiency changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency/(bits/mJ)')
    plt.legend()

    fig1_name = 'NetworkEnergyEfficiency_plot.png'
    plt.savefig(os.path.join(MAB_Config.result_folder_path, fig1_name))

    '''Network PDR'''
    plt.figure()
    plt.plot(x, MAB_Config.Network_PDR, '-', color='b')
    plt.title('Network PDR changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network PDR')

    fig2_name = 'Network_PDR_plot.png'
    plt.savefig(os.path.join(MAB_Config.result_folder_path, fig2_name))

    '''Network Throughput'''
    plt.figure()
    plt.plot(x, MAB_Config.Network_Throughput, '-', color='b')
    plt.title('Network throughput changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network Throughput')

    fig3_name = 'Network_Throughput.png'
    plt.savefig(os.path.join(MAB_Config.result_folder_path, fig3_name))

def result_record(PDR, EE):
    result_file = "Result.txt"
    result_file_path = os.path.join(MAB_Config.result_folder_path, result_file)
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
    

        


