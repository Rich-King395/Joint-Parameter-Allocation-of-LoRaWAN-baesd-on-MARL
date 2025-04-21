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
def MACMAB_run(nodes):
    # set_seed(random_seed)

    for episode in range(MACMAB_Config.sf_num_episode):
        MACMAB_train(nodes, episode)

    plot()
    
    MACMAB_Config.sf_train_flag = 1

    MACMAB_TP_allocation(nodes)
        
    MACMAB_eval(nodes)

    result_record(MACMAB_Config.NetPDR, MACMAB_Config.NetEnergyEfficiency)

''' 每一幕的数据包参数配置和所有智能体基础臂的期望奖励更新 '''
def MACMAB_train(nodes, episode):
    
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

        node.tp_index = 6
        node.sf_index = random.randint(0,5)
        node.fre_index = 0

        # node.tp_index = 6
        # node.sf_index = node.agent.actions_choose()
            
        # print("节点", node.id, "在第", episode, "幕选择的SF+BW为:", node.agent.sf_bw_arms[node.sf_bw_index],"选择的TP为:", node.agent.tp_arms[node.tp_index])
        
        ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
        env.process(MACMAB_transmit(env,node))

    # active_processes = len(env._queue)
    # print(f"当前环境中运行的进程数: {active_processes}")

    env.run(until=MACMAB_Config.eposide_duration)

    sumSent = 0
    sumSec = 0
    ''' 一幕结束对所有智能体期望奖励进行更新 '''
    for node in nodes:
        node.PDR = float((node.packetrec)/(node.sent))
        # print("节点", node.id, "在第", episode, "幕发送的包为",node.sent)
        ParameterConfig.PDRPerNode.append(node.PDR)
        node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
        ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
        node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
        ParameterConfig.ThroughputPerNode.append(node.Throughput)
        sumSent += node.sent
        sumSec += node.packetrec
        
        '''SF奖励'''
        node.agent.reward_SF = node.PDR

        node.agent.Expected_Reward_Update(node.sf_index)
    
    NetPDR = float(sumSec / sumSent) 
    NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    MACMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MACMAB_Config.Network_PDR.append(NetPDR)
    MACMAB_Config.Network_Throughput.append(NetThroughput)
    
    # print(f"episode={episode}, sumSent={sumSent}, ParameterConfig.sentPackets={len(ParameterConfig.sentPackets)},\
    #     sumSec={sumSec},ParameterConfig.recPackets={len(ParameterConfig.recPackets)}")

    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")

def MACMAB_TP_allocation(nodes):
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

        # print("节点", node.id, "在第", episode, "幕选择的SF+BW为:", node.agent.sf_bw_arms[node.sf_bw_index],"选择的TP为:", node.agent.tp_arms[node.tp_index])
        ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
        env.process(MACMAB_transmit(env,node))

    env.run(until=MACMAB_Config.tp_duration)

    ''' 计算每个节点在上一个发送幕中数据包的平均RSSI '''
    average_rssi = [sum(rssi_list) / len(rssi_list) if rssi_list else None for rssi_list in MACMAB_Config.node_rssi_list]

    print("average_rssi:", average_rssi)

    ''' 节点TP重新配置 '''
    for node in nodes:
        node_minisensi = myPacket.GetReceiveSensitivity(SF[node.sf_index],Bandwidth[node.bw_index])
        delta_Power = average_rssi[node.id] - node_minisensi
        if delta_Power > 0:
            theoretical_tp = Transmission_Power[node.tp_index] - delta_Power
            new_tp = min((tp for tp in Transmission_Power if tp > theoretical_tp), default=Transmission_Power[-1])
            if new_tp < 8:
                node.tp_index = np.where(Transmission_Power == new_tp)[0] + 3
            else:
                node.tp_index = np.where(Transmission_Power == new_tp)[0]

'''训练结束后, 测试训练效果'''
def MACMAB_eval(nodes):
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
        env.process(MACMAB_transmit(env,node))

    env.run(until=MACMAB_Config.eval_duration)

    MACMAB_Config.NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    MACMAB_Config.NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    MACMAB_Config.NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    print("Evaluation结束")
    print(f"Evaluation得到的网络指标为: PDR={MACMAB_Config.NetPDR*100:.2f}, Network EE={MACMAB_Config.NetEnergyEfficiency:.2f}, Network Throughput={MACMAB_Config.NetThroughput:.2f},")

def MACMAB_transmit(env,node):
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
                if (checkcollision(node.packet[bs])==1):    
                    node.packet[bs].collided = 1
                else:
                    node.packet[bs].collided = 0
                packetSeq += (bs+1)
                ParameterConfig.packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
                if MACMAB_Config.sf_train_flag == 1:
                    MACMAB_Config.node_rssi_list[node.id].append(node.packet[bs].RSSI)
                
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
                        ParameterConfig.recPackets.append(node.packet[bs].seqNr)
                        ParameterConfig.RecPacketSize += node.packet[bs].PS
                        node.RecPacketSize += node.packet[bs].PS
                    else:
                        # now need to check for right BS
                        if (node.bs.id == bs):
                            packetsRecBS[bs].append(node.packet[bs].seqNr)
                            node.packetrec += 1
                            ParameterConfig.recPackets.append(node.packet[bs].seqNr)
                            ParameterConfig.RecPacketSize += node.packet[bs].PS
                            node.RecPacketSize += node.packet[bs].PS
                    # recPackets is a global list of received packets
                    # not updated for multiple networks        
                else:
                    ParameterConfig.collidedPackets.append(node.packet[bs].seqNr)
                    node.packetloss += 1

            # print("num of total received packets",len(ParameterConfig.recPackets))
        
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

        PacketPara.sf = SF[node.sf_index]
        PacketPara.bw = Bandwidth[node.bw_index]
        PacketPara.fre = Carrier_Frequency[node.fre_index]
        PacketPara.tp = Transmission_Power[node.tp_index]                    
        packet = myPacket(node.id, PacketPara, i)
        node.packet.append(packet)

def reset_simulation_stats():
    """重置仿真统计量，确保每次动作测试独立"""
    ParameterConfig.packetsAtBS = [[] for _ in range(nrBS)]
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

def plot():
    base_result_folder_path = f"/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MACMAB/results"
    subfolder_name = f"{nrNodes}_nodes_{radius}_m"
    MACMAB_Config.result_folder_path = os.path.join(base_result_folder_path, subfolder_name)

    # 如果文件夹不存在则创建
    if not os.path.exists(MACMAB_Config.result_folder_path):
        os.makedirs(MACMAB_Config.result_folder_path)

    '''Network Energy Efficiency'''
    plt.figure()
    x = range(1,MACMAB_Config.sf_num_episode+1)
    plt.plot(x, MACMAB_Config.NetworkEnergyEfficiency, label='NetworkEnergyEfficiency', color='blue')

    plt.title('Network Energy Efficiency changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency/(bits/mJ)')
    plt.legend()

    fig1_name = 'NetworkEnergyEfficiency_plot.png'
    plt.savefig(os.path.join(MACMAB_Config.result_folder_path, fig1_name))

    '''Network PDR'''
    plt.figure()
    plt.plot(x, MACMAB_Config.Network_PDR, '-', color='b')
    plt.title('Network PDR changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network PDR')

    fig2_name = 'Network_PDR_plot.png'
    plt.savefig(os.path.join(MACMAB_Config.result_folder_path, fig2_name))

    '''Network Throughput'''
    plt.figure()
    plt.plot(x, MACMAB_Config.Network_Throughput, '-', color='b')
    plt.title('Network throughput changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network Throughput')

    fig3_name = 'Network_Throughput.png'
    plt.savefig(os.path.join(MACMAB_Config.result_folder_path, fig3_name))

def result_record(PDR, EE):
    result_file = "Result.txt"
    result_file_path = os.path.join(MACMAB_Config.result_folder_path, result_file)
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
        file.write('PDR={:.3f} %,'.format(MACMAB_Config.Network_PDR[-1]*100))
        file.write('Network EE={:.3f} bits/mJ\n'.format(MACMAB_Config.NetworkEnergyEfficiency[-1]))
        file.write('Results after power control:\n')
        file.write('PDR={:.3f} %,'.format(PDR*100))
        file.write('Network EE={:.3f} bits/mJ\n'.format(EE))
        



