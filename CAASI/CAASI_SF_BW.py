#------------Channel Allocation and Action Space Innitialization-------------#
import ParameterConfig
from ParameterConfig import *
from Propagation import *
from Packet import myPacket

def CAASI_run(nodes):
    '''Channel Allocation'''
    global env
    for i in range(CASSI_Config.n_period):
        for node in nodes:
            if (numChannel * i) <= node.id <= (numChannel * i + 7):
                CASSI_Config.nodes_transmit.append(node)
                env.process(CAASI_transmit(env, node))

        for j in range(numChannel):
            for node in CASSI_Config.nodes_transmit:
                '''SF = 12, BW = 125kHz'''
                node.sf_index = 5
                node.bw_index = 0
                node.fre_index = (node.id + j) % numChannel
            # initialize simulation environment current time for each episode
            env.run(until=CASSI_Config.CF_time)
            env = simpy.Environment()                        

        CASSI_Config.nodes_transmit = []
    
    # 计算每个节点的平均 RSSI
    Node_RSSI = sorted([
        (node_id, sum(sum(ch) for ch in CASSI_Config.rssi_measurements[node_id]) / 
            sum(len(ch) for ch in CASSI_Config.rssi_measurements[node_id]) if sum(len(ch) for ch in CASSI_Config.rssi_measurements[node_id]) > 0 else float('inf'))
        for node_id in range(nrNodes)
    ], key=lambda x: x[1])  # 升序排序

    # 计算每个信道的平均 RSSI
    Channel_RSSI = sorted([
        (channel_id, sum(rssi for node in CASSI_Config.rssi_measurements for rssi in node[channel_id]) / 
            sum(len(node[channel_id]) for node in CASSI_Config.rssi_measurements) if sum(len(node[channel_id]) for node in CASSI_Config.rssi_measurements) > 0 else float('-inf'))
        for channel_id in range(numChannel)
    ], key=lambda x: x[1], reverse=True)  # 降序排序

    # 打印结果
    print("Node RSSI (sorted in ascending order):", Node_RSSI)
    print("Channel RSSI (sorted in descending order):", Channel_RSSI)

    '''将排序后的节点分为8组并分配信道'''
    # 计算每组的大小
    num_groups = 8
    group_size = nrNodes // num_groups  
    remainder = nrNodes % num_groups    
    node_groups = []
    index = 0
    for i in range(num_groups):
        size = group_size + (1 if i < remainder else 0)  # 余数分布到前 remainder 组
        node_groups.append(Node_RSSI[index:index + size])
        index += size

    for index, node_group in enumerate(node_groups):
        for nodeid_rssi in node_group:
            nodes[nodeid_rssi[0]].fre_index = Channel_RSSI[index][0]
            # print("节点",nodes[nodeid_rssi[0]].id,"分配到的信道",Channel_RSSI[index][0])
    
    '''Action Space Initialization'''
    # 计算最多需要多少轮（取决于最长的 node_groups）
    max_rounds = max(len(group) for group in node_groups)
    
    for round_num in range(max_rounds):
        for group in node_groups:
            if round_num < len(group):  # 该组还有节点可用
                CASSI_Config.nodes_transmit.append(nodes[group[round_num][0]])


        for k in range(num_sf_bw):
            for i in range(0,nrBS):
                ParameterConfig.packetsAtBS.append([])
            for node in CASSI_Config.nodes_transmit:
                node.sf_index = sf_bw_list[k][0]
                node.bw_index = sf_bw_list[k][1]
                
                env.process(CAASI_transmit(env, node))
            
            env.run(until=CASSI_Config.SF_BW_time)
            
            for node in CASSI_Config.nodes_transmit:
                node.PDR = (node.packetrec / node.sent) * 100
                '''剔除节点不合理的SF+BW组合(数据包全部丢失)'''
                # print("节点", node.id, "在SF+BW组合为",sf_bw_list[k],"下的PDR为:",node.PDR)
                if node.PDR < CASSI_Config.sf_bw_PDR_thres:
                    if sf_bw_list[k] in node.agent.sf_bw_arms:
                        node.agent.sf_bw_arms.remove(sf_bw_list[k])
                    
                node.sent = 0
                node.packetloss = 0
                node.packetrec = 0
                node.PDR = 0
                node.collided = 0

            env = simpy.Environment()       
        
        CASSI_Config.nodes_transmit = []
    

    '''动作空间初始化后对Q值表和计数表重新初始化'''
    for node in nodes:
        node.agent.K_SF_BW = len(node.agent.sf_bw_arms)
        node.agent.Q_SF_BW = np.zeros(node.agent.K_SF_BW, dtype=float)
        node.agent.counts_SF_BW = np.zeros(node.agent.K_SF_BW)

                                            
def CAASI_transmit(env, node):    
    while True:
        yield env.timeout(random.expovariate(1.0 / float(node.period)))
        
        global packetSeq
        CAASI_Generate_Packet(node)
        node.sent += nrBS  # 节点发送数据包数

        for bs in range(nrBS):
            if node in ParameterConfig.packetsAtBS[bs]:
                pass
            else:
                if checkcollision(node.packet[bs]) == 1:    
                    node.packet[bs].collided = 1
                    node.collided += 1
                else:
                    node.packet[bs].collided = 0
                packetSeq += (bs+1)
                ParameterConfig.packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
                CASSI_Config.rssi_measurements[node.id][node.fre_index].append(node.packet[bs].RSSI)

        print("packetsAtBS的长度:", len(ParameterConfig.packetsAtBS[0]))

        for bs in range(nrBS):
            if node.packet[bs].lost:
                node.packetloss += 1                
            else:
                if node.packet[bs].collided == 0:
                    if nrNetworks == 1:
                        node.packetrec += 1
                        # print("node.packet[bs].RSSI:", node.packet[bs].RSSI)                        
                    else:
                        if node.bs.id == bs:
                            node.packetrec += 1
                else:
                    node.packetloss += 1

        for bs in range(nrBS):                    
            if node in ParameterConfig.packetsAtBS[bs]:
                ParameterConfig.packetsAtBS[bs].remove(node)
                node.packet[bs].collided = 0
                # node.packet[bs].lost = False  

        yield env.timeout(node.packet[0].rectime)

        

# node generate "virtul" packets for each gateway
def CAASI_Generate_Packet(node):
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
        PacketPara.tp = Transmission_Power[6] # TP = 14dBm                    
        packet = myPacket(node.id, PacketPara, i)
        node.packet.append(packet)



