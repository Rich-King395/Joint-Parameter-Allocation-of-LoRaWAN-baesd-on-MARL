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
import random
def ILCMAB_run(nodes):
    # set_seed(random_seed)

    for episode in range(MACMAB_Config.num_episode):
        ILCMAB_train(nodes, episode)

    plot("/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/ILCMAB_SF/results")
               
    # MACMAB_eval(nodes)

''' 每一幕的数据包参数配置和所有智能体基础臂的期望奖励更新 '''
def ILCMAB_train(nodes, episode):
    
    # initialize simulation environment current time for each episode
    env = simpy.Environment()

    ''' initialize the environment at the beginning of episode '''
    # nodes sent to each GW
    for i in range(0,nrBS):
        ParameterConfig.packetsAtBS.append([]) 
    # Packets' sequence number received by each GW
    ParameterConfig.packetsRecBS = [] 


    # list of sent packets 
    ParameterConfig.sentPackets = [] 
    # list of received packets
    ParameterConfig.recPackets = []
    # list of collided packets
    ParameterConfig.collidedPackets=[] 
    # list of lost packets
    ParameterConfig.lostPackets = [] 

    ParameterConfig.RecPacketSize = 0
    ParameterConfig.TotalPacketSize = 0
    ParameterConfig.TotalPacketAirtime = 0
    ParameterConfig.TotalEnergyConsumption = 0
    
    ParameterConfig.PDRPerNode = []
    ParameterConfig.EnergyEfficiencyPerNode = []
    ParameterConfig.ThroughputPerNode = []


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
        env.process(ILCMAB_transmit(env,node))

    # active_processes = len(env._queue)
    # print(f"当前环境中运行的进程数: {active_processes}")

    env.run(until=MACMAB_Config.eposide_duration)

    NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    MACMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MACMAB_Config.Network_PDR.append(NetPDR)
    MACMAB_Config.Network_Throughput.append(NetThroughput)

    for node in nodes:
        node.PDR = float((node.packetrec)/(node.sent))
        # print("节点", node.id, "在第", episode, "幕发送的包为",node.sent)
        ParameterConfig.PDRPerNode.append(node.PDR)
        node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
        ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
        node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
        ParameterConfig.ThroughputPerNode.append(node.Throughput)
       
    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")


# '''训练结束后, 测试训练效果'''
# def MACMAB_eval(nodes):

def ILCMAB_transmit(env,node):
    while True:
        # set_seed(random_seed)
        yield env.timeout(random.expovariate(1.0/float(node.period)))
                     
        global packetSeq
        CMAB_Generate_Packet(node)
        node.sent = node.sent + nrBS # number of packets sent by the node

        for bs in range(0, nrBS):
            if node in ParameterConfig.packetsAtBS[bs]:
                 pass
            else:
                # adding packet if no collision
                if checkcollision(node.packet[bs])==1:    
                    node.packet[bs].collided = 1
                else:
                    node.packet[bs].collided = 0
                packetSeq += (bs+1)
                ParameterConfig.packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
                # print("node.packet[bs].seqNr:",node.packet[bs].seqNr)
            
            node.EnergyConsumption += node.packet[bs].tx_energy
            node.TotalPacketAirtime += float(node.packet[bs].rectime / 1000) 

            ParameterConfig.sentPackets.append(node.packet[bs].seqNr)   
            ParameterConfig.TotalPacketSize += node.packet[bs].PS
            ParameterConfig.TotalEnergyConsumption += node.packet[bs].tx_energy
            ParameterConfig.TotalPacketAirtime += float(node.packet[bs].rectime / 1000)   

        # print("packetsAtBS的长度:", len(ParameterConfig.packetsAtBS[0])) 
               
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
                node.agent.reward_SF = -1 # SF基础臂的奖励
                # reward_towards_EE = 1 - float(Transmission_Power[node.tp_index]/TP_SUM)
                # node.agent.reward_TP = -1 + reward_towards_EE # TP基础臂的奖励
                node.agent.reward_TP = -1 # TP基础臂的奖励
            else: 
                '''数据包被成功接收给正奖励'''
                reward_towards_small_SF =1 - float((2^(SF[node.sf_index]))/SF_SUM)
                # node.agent.reward_SF = 1 + reward_towards_small_SF # SF基础臂的奖励
                node.agent.reward_SF = 1
                
                reward_towards_EE = 1 - float(Transmission_Power[node.tp_index]/Transmission_Power[6])
                node.agent.reward_TP = 1 + reward_towards_EE # TP基础臂的奖励
        
        node.agent.Expected_Reward_Update(node.sf_index, node.tp_index)

        # complete packet has been received by base station
        # can remove it for next transmission
        for bs in range(0, nrBS):                    
            if node in ParameterConfig.packetsAtBS[bs]:
                ParameterConfig.packetsAtBS[bs].remove(node)
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

        node.sf_index, node.tp_index = node.agent.actions_choose()
        PacketPara.sf = SF[node.sf_index]
        PacketPara.bw = Bandwidth[node.bw_index]
        PacketPara.fre = Carrier_Frequency[node.fre_index]
        PacketPara.tp = Transmission_Power[node.tp_index]                    
        packet = myPacket(node.id, PacketPara, i)
        node.packet.append(packet)


def plot(result_folder_path):
    '''Network Energy Efficiency'''
    plt.figure()
    x = range(1,MACMAB_Config.num_episode+1)
    plt.plot(x, MACMAB_Config.NetworkEnergyEfficiency, label='NetworkEnergyEfficiency', color='blue')

    plt.title('Network Energy Efficiency changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency/(bits/mJ)')
    plt.legend()

    fig1_name = 'NetworkEnergyEfficiency_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig1_name))

    '''Network PDR'''
    plt.figure()
    plt.plot(x, MACMAB_Config.Network_PDR, '-', color='b')
    plt.title('Network PDR changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network PDR')

    fig2_name = 'Network_PDR_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig2_name))

    '''Network Throughput'''
    plt.figure()
    plt.plot(x, MACMAB_Config.Network_Throughput, '-', color='b')
    plt.title('Network throughput changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network Throughput')

    fig3_name = 'Network_Throughput.png'
    plt.savefig(os.path.join(result_folder_path, fig3_name))