import os
import matplotlib.pyplot as plt
import random
from ParameterConfig import *
import ParameterConfig
from Node import *
from Gateway import *
import random
from itertools import product
from CoCMAB.CoAgent import CUCB
def CoCMAB_run(nodes):
    # set_seed(random_seed)

    '''开始训练前先对联合动作空间进行初始化, 并初始化CoAgent'''
    for group_idx, group in enumerate(CASSI_Config.node_groups): 
        group_node_ids = [node_id_rssi[0] for node_id_rssi in group]  # 提取组内所有节点的 ID
        
        # 获取该组所有节点的 SF 动作空间
        group_sf_arms = [nodes[node_id].agent.sf_arms for node_id in group_node_ids]
        
        # 计算联合动作（笛卡尔积）
        joint_action = list(product(*group_sf_arms))  # 所有可能的 SF 组合

        # 创建 CoAgent，并赋予 id
        CoAgent = CUCB(joint_action)
        CoAgent.id = group_idx  

        print("CoAgent", CoAgent.id, "的动作空间为：", CoAgent.sf_arms)
        
         # 存储联合动作和节点顺序
        CoCMAB_Config.joint_actions.append(joint_action)
        CoCMAB_Config.joint_action_nodes.append(group_node_ids)

        # 存入 CoCMAB_Config.CoAgents
        CoCMAB_Config.CoAgents.append(CoAgent)
    
    for i in range(2):
        Intialize(nodes)

    print("期望奖励初始化阶段结束......")
    print("开始训练......")

    for episode in range(CoCMAB_Config.sf_num_episode):
        CoCMAB_SF_train(nodes, episode)
    
    print("联合SF分配训练结束......")
    print("开始TP分配训练......")

    CoCMAB_Config.sf_train_flag = 1

    for episode in range(CoCMAB_Config.tp_num_episode):
        CoCMAB_TP_train(nodes, episode)
    
    plot("/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/CoCMAB/results")
               
    # MACMAB_eval(nodes)

'''在训练开始前, 对CoAgent的所有动作至少执行一次, 得到初始期望奖励'''
def Intialize(nodes):
    Initialize_Episode = max(len(CoAgent.sf_arms) for CoAgent in CoCMAB_Config.CoAgents) if CoCMAB_Config.CoAgents else 0
    print("初始化训练幕数为：", Initialize_Episode)
    for k in range(Initialize_Episode):
        env = simpy.Environment()
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

            '''初始化阶段TP保持为最大'''
            node.tp_index = 6

            ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
            env.process(CoCMAB_transmit(env,node))

        '''CoAgent进行联合SF选择'''
        for CoAgent in CoCMAB_Config.CoAgents:
            if k < len(CoAgent.sf_arms):
                CoAgent.sf_index = k
                joint_action_choose = CoAgent.sf_arms[k]
                CoAgent.counts_SF[k] += 1
                CoAgent.total_count += 1
            # print("CoAgent",CoAgent.id, "joint_action_choose:", CoAgent.sf_index)
            else:
                 joint_action_choose = random.choice(CoAgent.sf_arms)
                 CoAgent.sf_index = CoAgent.sf_arms.index(joint_action_choose)
                 CoAgent.counts_SF[CoAgent.sf_index] += 1
                 CoAgent.total_count += 1


            '''节点与联合动作进行映射'''
            action_mapping = dict(zip(CoCMAB_Config.joint_action_nodes[CoAgent.id], joint_action_choose))
            for node_id, sf_index in action_mapping.items():
                nodes[node_id].sf_index = sf_index 
    
        # active_processes = len(env._queue)
        # print(f"当前环境中运行的进程数: {active_processes}")

        env.run(until=MACMAB_Config.eposide_duration)

        NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
        NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
        NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

        MACMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
        MACMAB_Config.Network_PDR.append(NetPDR)
        MACMAB_Config.Network_Throughput.append(NetThroughput)

        ''' 一幕结束对所有智能体期望奖励进行更新 '''
        for node in nodes:
            node.PDR = float((node.packetrec)/(node.sent))
            # print("节点", node.id, "在第", episode, "幕发送的包为",node.sent)
            ParameterConfig.PDRPerNode.append(node.PDR)
            node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
            ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
            node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
            ParameterConfig.ThroughputPerNode.append(node.Throughput)
        
        for CoAgent in CoCMAB_Config.CoAgents:
            group_packet_sent = 0
            group_packet_rec = 0
            for node_id in CoCMAB_Config.joint_action_nodes[CoAgent.id]:
                group_packet_sent += nodes[node_id].sent
                group_packet_rec += nodes[node_id].packetrec
            group_PDR = float(group_packet_rec / group_packet_sent)
            CoAgent.reward_SF = group_PDR
            CoAgent.Expected_Reward_Update(CoAgent.sf_index)
    

''' 每一幕的数据包参数配置和所有智能体基础臂的期望奖励更新 '''
def CoCMAB_SF_train(nodes, episode):
    
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

        '''SF训练阶段TP保持为最大'''
        node.tp_index = 6

        ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
        env.process(CoCMAB_transmit(env,node))

    '''CoAgent进行联合SF选择'''
    for CoAgent in CoCMAB_Config.CoAgents:
        CoAgent.sf_index, joint_action_choose = CoAgent.actions_choose()
        # print("CoAgent",CoAgent.id, "joint_action_choose:", joint_action_choose)
        '''节点与联合动作进行映射'''
        action_mapping = dict(zip(CoCMAB_Config.joint_action_nodes[CoAgent.id], joint_action_choose))
        for node_id, sf_index in action_mapping.items():
            nodes[node_id].sf_index = sf_index 

    # active_processes = len(env._queue)
    # print(f"当前环境中运行的进程数: {active_processes}")

    env.run(until=MACMAB_Config.eposide_duration)

    NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    MACMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MACMAB_Config.Network_PDR.append(NetPDR)
    MACMAB_Config.Network_Throughput.append(NetThroughput)

    ''' 一幕结束对所有智能体期望奖励进行更新 '''
    for node in nodes:
        node.PDR = float((node.packetrec)/(node.sent))
        # print("节点", node.id, "在第", episode, "幕发送的包为",node.sent)
        ParameterConfig.PDRPerNode.append(node.PDR)
        node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
        ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
        node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
        ParameterConfig.ThroughputPerNode.append(node.Throughput)
       
    for CoAgent in CoCMAB_Config.CoAgents:
        group_packet_sent = 0
        group_packet_rec = 0
        for node_id in CoCMAB_Config.joint_action_nodes[CoAgent.id]:
            group_packet_sent += nodes[node_id].sent
            group_packet_rec += nodes[node_id].packetrec
        group_PDR = float(group_packet_rec / group_packet_sent)
        CoAgent.reward_SF = group_PDR
        CoAgent.Expected_Reward_Update(CoAgent.sf_index)

    print(f"episode={episode}, PDR={NetPDR*100:.2f}, Network EE={NetEnergyEfficiency:.2f}, Network Throughput={NetThroughput:.2f},")

def CoCMAB_TP_train(nodes, episode):
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

        # print("节点", node.id, "在第", episode, "幕选择的SF+BW为:", node.agent.sf_bw_arms[node.sf_bw_index],"选择的TP为:", node.agent.tp_arms[node.tp_index])
        ''' 在仿真开始前，为每个节点创建数据包传输进程 '''
        env.process(CoCMAB_transmit(env,node))

    env.run(until=MACMAB_Config.eposide_duration)

    NetPDR = float(len(ParameterConfig.recPackets)/len(ParameterConfig.sentPackets)) 
    NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
    NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

    MACMAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
    MACMAB_Config.Network_PDR.append(NetPDR)
    MACMAB_Config.Network_Throughput.append(NetThroughput)

    ''' 一幕结束对所有智能体期望奖励进行更新 '''
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

def CoCMAB_transmit(env,node):
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
        
        if CoCMAB_Config.sf_train_flag == 1:
            '''节点每次数据包传输都能获得奖励，并对智能体的期望奖励进行更新'''
            reward_towards_EE = 1 - float(Transmission_Power[node.tp_index]/Transmission_Power[6])
            for bs in range(0, nrBS):
                if node.packet[bs].lost == True or node.packet[bs].collided == 1: 
                    '''数据包丢失给负奖励'''
                    node.agent.reward_TP = -1 # TP基础臂的奖励
                else: 
                    '''数据包被成功接收给正奖励'''
                    node.agent.reward_TP = 1 + reward_towards_EE # TP基础臂的奖励

            node.agent.Expected_Reward_Update(node.tp_index)

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


        PacketPara.sf = SF[node.sf_index]
        PacketPara.bw = Bandwidth[node.bw_index]
        PacketPara.fre = Carrier_Frequency[node.fre_index]
        if CoCMAB_Config.sf_train_flag == 1:
            '''节点上的TP_MAB进行功率选择'''
            node.tp_index = node.agent.actions_choose()
            PacketPara.tp = Transmission_Power[node.tp_index]
        else:
            PacketPara.tp = Transmission_Power[6]                    
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