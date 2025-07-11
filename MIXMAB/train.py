import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import *
from Gateway import *
from plot import *
import random

global first_record_call 
first_record_call = True
def MIXMAB_run(nodes):
    # set_seed(random_seed)
    for episode in range(MIXMAB_Config.num_episode):
        ParameterConfig.global_episode = episode
        MIXMAB_train(nodes, episode)

    training_process(MIXMAB_Config)

    MIXMAB_eval(nodes)

    result_record(MIXMAB_Config.NetPDR, MIXMAB_Config.NetEnergyEfficiency)              

''' 每一幕的数据包参数配置和所有智能体基础臂的期望奖励更新 '''
def MIXMAB_train(nodes, episode):
    
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
        env.process(MIXMAB_transmit(env,node))
        # env.process(transmit(env,node))

    # active_processes = len(env._queue)
    # print(f"当前环境中运行的进程数: {active_processes}")

    env.run(until=MIXMAB_Config.eposide_duration)

    sumRec = 0
    sumSent = 0
    for node in nodes:
        node.PDR = float((node.packetrec)/(node.sent))
        # print("节点", node.id, "在第", episode, "幕发送的包为",node.sent)
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

    MIXMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MIXMAB_Config.Network_PDR.append(NetPDR)
    MIXMAB_Config.Network_Throughput.append(NetThroughput)

    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")
    record(episode, NetPDR, NetEnergyEfficiency)

'''训练结束后, 测试训练效果'''
def MIXMAB_eval(nodes):
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
        env.process(MIXMAB_transmit(env,node))

    env.run(until=MIXMAB_Config.eval_duration)

    for node in nodes:
        # print("node.id:",node.id,"SF:",SF[node.sf_index])
        sf_distribute[node.sf_index] += 1
        tp_distribute[node.tp_index] += 1
        fre_distribute[node.fre_index] += 1

    MIXMAB_Config.NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    MIXMAB_Config.NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    MIXMAB_Config.NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)
    
    sf_distribution(MIXMAB_Config.result_folder_path)
    tp_distribution(MIXMAB_Config.result_folder_path)

    print("Evaluation结束")
    print(f"Evaluation得到的网络指标为: PDR={MIXMAB_Config.NetPDR*100:.2f}, Network EE={MIXMAB_Config.NetEnergyEfficiency:.2f}, Network Throughput={MIXMAB_Config.NetThroughput:.2f},")



def MIXMAB_transmit(env,node):
    while True:
        # set_seed(random_seed)
        yield env.timeout(random.expovariate(1.0/float(node.period)))
                     
        global packetSeq
        CMAB_Generate_Packet(node)
        node.sent = node.sent + nrBS # number of packets sent by the node

        for bs in range(0, nrBS):
            if node in packetsAtBS[bs]:
                 pass
            else:
                # adding packet if no collision
                if checkcollision(node.packet[bs])==1:    
                    node.packet[bs].collided = 1
                else:
                    node.packet[bs].collided = 0
                packetSeq += (bs+1)
                packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
                # print("node.packet[bs].seqNr:",node.packet[bs].seqNr)
            
            node.EnergyConsumption += node.packet[bs].tx_energy
            node.TotalPacketAirtime += float(node.packet[bs].rectime / 1000) 

            ParameterConfig.sentPackets.append(node.packet[bs].seqNr)   
            ParameterConfig.TotalPacketSize += node.packet[bs].PS
            ParameterConfig.TotalEnergyConsumption += node.packet[bs].tx_energy
            ParameterConfig.TotalPacketAirtime += float(node.packet[bs].rectime / 1000)   

        # print("packetsAtBS的长度:", len(packetsAtBS[0])) 
               
        # ket time on air   
        yield env.timeout(node.packet[0].rectime)

        # if packet did not collide, add it in list of received packets
        # unless it is already in
        for bs in range(0, nrBS):
            if node.packet[bs].lost:
                ParameterConfig.lostPackets.append(node.packet[bs].seqNr)
                node.packetloss += 1
                # print("节点",node.id, "丢失的数据包数为：",node.packetloss)
            else:
                if node.packet[bs].collided == 0:
                    if (nrNetworks == 1):
                        packetsRecBS[bs].append(node.packet[bs].seqNr)
                        node.packetrec += 1
                    else:
                        # now need to check for right BS
                        if (node.bs.id == bs):
                            packetsRecBS[bs].append(node.packet[bs].seqNr)
                            node.packetrec += 1
                    # recPackets is a global list of received packets
                    # not updated for multiple networks        
                    if (ParameterConfig.recPackets):
                        if (ParameterConfig.recPackets[-1] != node.packet[bs].seqNr):
                            ParameterConfig.recPackets.append(node.packet[bs].seqNr)
                            # print("num of total received packets",len(ParameterConfig.recPackets))      
                            ParameterConfig.RecPacketSize += node.packet[bs].PS
                            node.RecPacketSize += node.packet[bs].PS
                    else:
                        ParameterConfig.recPackets.append(node.packet[bs].seqNr)
                        ParameterConfig.RecPacketSize += node.packet[bs].PS
                        node.RecPacketSize += node.packet[bs].PS
                else:
                    ParameterConfig.collidedPackets.append(node.packet[bs].seqNr)
                    node.packetloss += 1

            # print("num of total received packets",len(ParameterConfig.recPackets))
        
        '''节点每次数据包传输都能获得奖励，并对智能体的期望奖励进行更新'''
        for bs in range(0, nrBS):
            if node.packet[bs].lost == True or node.packet[bs].collided == 1: 
                '''数据包丢失给负奖励'''
                node.agent.reward = 0
            else: 
                '''数据包被成功接收给正奖励''' 
                node.agent.reward = 1
        
        node.agent.Probability_Weight_Update(node.arm_index)
        

        # complete packet has been received by base station
        # can remove it for next transmission
        for bs in range(0, nrBS):                    
            if node in packetsAtBS[bs]:
                packetsAtBS[bs].remove(node)
                node.packet[bs].collided = 0
                node.packet[bs].lost = False

        
# node generate "virtul" packets for each gateway
def CMAB_Generate_Packet(node):
    node.packet = []
    for i in range(0,nrBS):
        # d = get_distance(self.x,self.y,bs[i]) # distance between node and gateway
        # self.dist.append(d)
        ''' ''' 
        if node.Generate_Packet_Flag == 0:
            PacketPara = LoRaParameters()
            node.Generate_Packet_Flag == 1

        node.arm_index = node.agent.actions_choose()
        PacketPara.sf = Parameter_Configurations[node.arm_index][1]
        PacketPara.bw = Bandwidth[node.bw_index]
        PacketPara.fre =  Parameter_Configurations[node.arm_index][0]
        PacketPara.tp =  Parameter_Configurations[node.arm_index][2]  
        node.fre_index = np.where(Carrier_Frequency == PacketPara.fre)[0][0]          
        node.sf_index = np.where(SF == PacketPara.sf)[0][0] 
        node.tp_index = np.where(Transmission_Power == PacketPara.tp)[0][0] 
        packet = myPacket(node.id, PacketPara, i)
        node.packet.append(packet)

def reset_simulation_stats():
    """重置仿真统计量，确保每次动作测试独立"""
    packetsAtBS = [[] for _ in range(nrBS)]
    ParameterConfig.packetsRecBS = []
    ParameterConfig.sentPackets = []
    ParameterConfig.recPackets = []
    ParameterConfig.collidedPackets = []
    ParameterConfig.lostPackets = []
    ParameterConfig.RecPacketSize = 0
    ParameterConfig.TotalPacketSize = 0
    ParameterConfig.TotalEnergyConsumption = 0
    ParameterConfig.TotalPacketAirtime = 0

    ParameterConfig.PDRPerNode = []
    ParameterConfig.EnergyEfficiencyPerNode = []
    ParameterConfig.ThroughputPerNode = []

def record(episode, PDR, EE):
    base_result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MIXMAB/results"
    
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
    

    MIXMAB_Config.result_folder_path = os.path.join(base_result_folder_path, subfolder_name)

    if not os.path.exists(MIXMAB_Config.result_folder_path):
        os.makedirs(MIXMAB_Config.result_folder_path)
    
    record_file = "record.txt"
    record_file_path = os.path.join(MIXMAB_Config.result_folder_path, record_file)

    global first_record_call
    mode = 'w' if first_record_call else 'a'
    with open(record_file_path, mode) as file:
        file.write(f"episode={episode} PDR={PDR*100:.2f} EE={EE:.2f}\n")
    
    first_record_call = False  

def result_record(PDR, EE):
    result_file = "Result.txt"
    result_file_path = os.path.join(MIXMAB_Config.result_folder_path, result_file)
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
        file.write('Results after SF allocation before power control:\n')
        file.write('PDR={:.3f} %,'.format(MIXMAB_Config.Network_PDR[-1]*100))
        file.write('Network EE={:.3f} bits/mJ\n'.format(MIXMAB_Config.NetworkEnergyEfficiency[-1]))
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
