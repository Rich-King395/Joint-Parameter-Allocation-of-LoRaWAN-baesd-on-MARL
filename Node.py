#
# this function creates a node
# generate a node in the network with a random space
import matplotlib.pyplot as plt
import numpy as np
from ParameterConfig import *
import ParameterConfig
from Packet import myPacket
from Allocation import *
from MAB.Agent import *

class myNode:
    def __init__(self, id, x, y, period, myBS):
        self.bs = myBS # the BS, which the node needs to send packets to
        self.id = id
        self.period = period

        self.x = x
        self.y = y

        self.last_packet_rssi = 0 # the RSSI of the last packet sent by the node
        self.ADR_flag = 0 # the ADR flag of the node
        self.Generate_Packet_Flag = 0

        if allocation_method == "MAB":
            if MAB_Config.MAB_Variant == 0:
                self.agent = EpsilonGreedy()
            elif MAB_Config.MAB_Variant == 1:
                self.agent = DecayingEpsilonGreedy()
            elif MAB_Config.MAB_Variant == 2:
                self.agent = UCB(MAB_Config.coef)

            # self.agent = DecayingEpsilonGreedy() 
            # self.agent = EpsilonGreedy(MAB_Config.epsilon)

        # LoRa parameters the node used to send packets      
        self.sf_index = 0
        self.bw_index = 0
        self.fre_index = 0
        self.tp_index = 0
        self.sf_bw_index = 0

        self.packet = []
        self.dist = []

        # list of packets sent during update interval
        self.packets_interval = []

        # number of packets sent by the node
        self.sent = 0
        self.powerlost = 0
        self.collided = 0

        self.sent_interval = 0 # number of packets sent by the node during update interval
        self.rec_interval = 0 # number of sussccessfully-seceived packets sent by the node during update interval
        self.lost_interval = 0 # number of lost packets sent by the node during update interval

        self.PDR = 0 # packet delivery ratio of the node
        self.RecPacketSize = 0 # size of packets received by the node
        self.EnergyConsumption = 0 # energy consumption of the node
        self.EnergyEfficiency = 0 # energy efficiency of the node

        if allocation_type == "Global":
            myNode.Generate_Packet(self)

        # graphics for node
        global graphics
        if (graphics == 1):
            global ax
            if (self.bs.id == 0):
                    ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='blue'))
            if (self.bs.id == 1):
                    ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='red'))
            if (self.bs.id == 2):
                    ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='green'))
            if (self.bs.id == 3):
                    ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='brown'))
            if (self.bs.id == 4):
                    ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='orange'))
    
    # node generate "virtul" packets for each gateway
    def Generate_Packet(self):
        self.packet = []
        for i in range(0,nrBS):
            d = get_distance(self.x,self.y,bs[i]) # distance between node and gateway
            self.dist.append(d)
            if self.Generate_Packet_Flag == 0:
                PacketPara = LoRaParameters()
                self.Generate_Packet_Flag == 1
            if allocation_method == "random":
                 PacketPara.sf,PacketPara.bw,PacketPara.fre,PacketPara.tp = random_allocation()
            elif allocation_method == "ADR":
                 PacketPara.sf,PacketPara.bw,PacketPara.fre,PacketPara.tp = ADR(PacketPara,self.last_packet_rssi,self.ADR_flag,self.id)
            elif allocation_method == "Round Robin":
                 PacketPara.sf,PacketPara.bw,PacketPara.fre = round_robin_allocation(self.id)
            elif allocation_method == "MARL":                     
                 PacketPara.sf = SF[self.agent.action[0]]
            elif allocation_method == "MAB":
                 self.sf_index,self.bw_index,self.fre_index,self.tp_index = self.agent.actions_choose()
                 PacketPara.sf = SF[self.sf_index]
                 PacketPara.bw = Bandwidth[self.bw_index]
                 PacketPara.fre = Carrier_Frequency[self.fre_index]
                 PacketPara.tp = Transmission_Power[self.tp_index]
                 #PacketPara.tp = 14
            elif allocation_method == "Q-table":
                 self.sf_index,self.bw_index,self.fre_index = self.agent.actions_choose()
                 PacketPara.sf = SF[self.sf_index]
                 PacketPara.bw = Bandwidth[self.bw_index]
                 PacketPara.fre = Carrier_Frequency[self.fre_index]
            packet = myPacket(self.id, PacketPara, i)
            self.last_packet_rssi = checklost(packet,self.dist[i])
            # if self.id == 0:
            #     print("last_packet_rssi", self.last_packet_rssi)
            self.packet.append(packet)
            self.packets_interval.append(packet)
        # print('node %d' %id, "x", self.x, "y", self.y, "dist: ", self.dist, "my BS:", self.bs.id)

#   directional antenna
#   update RSSI depending on direction
#
    def updateRSSI(self):
        # print ("+++++++++uR node", self.id, " and bs ", self.bs.id) 
        # print ("node x,y", self.x, self.y)
        # print ("main-bs x,y", bs[self.bs.id].x, bs[self.bs.id].y)
        for i in range(0,len(self.packet)):
            # print ("rssi before", self.packet[i].RSSI)
            # print ("packet bs", self.packet[i].bs)
            # print ("packet bs x, y:", bs[self.packet[i].bs].x, bs[self.packet[i].bs].y)            
            if (self.bs.id == self.packet[i].bs): # node send packet to its main BS
                print ("packet to main bs, increase rssi ")
                self.packet[i].RSSI = self.packet[i].RSSI + dir_30
            else: # node send packet to other BS

                b1 = np.array([bs[self.bs.id].x, bs[self.bs.id].y]) # position of the main BS
                p = np.array([self.x, self.y]) # position of the node
                b2 = np.array([bs[self.packet[i].bs].x, bs[self.packet[i].bs].y]) # position of the sending BS

                ba = b1 - p # vector ba
                bc = b2 - p # vector bc
                print (ba)
                print (bc)

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.degrees(np.arccos(cosine_angle))

                print ("angle: ", angle)

                if (angle <= 30):
                    print ("rssi increase to other BS: 4")
                    self.packet[i].RSSI = self.packet[i].RSSI + dir_30
                elif angle <= 90:
                    print ("rssi increase to other BS: 2")
                    self.packet[i].RSSI = self.packet[i].RSSI + dir_90
                elif angle <= 150:
                    print ("rssi increase to other BS: -4")
                    self.packet[i].RSSI = self.packet[i].RSSI + dir_150
                else:
                    print ("rssi increase to other BS: -3")
                    self.packet[i].RSSI = self.packet[i].RSSI + dir_180
            print ("packet rssi after", self.packet[i].RSSI)

def get_distance(x,y,GW):
     dist = np.sqrt((x-GW.x)*(x-GW.x)+(y-GW.y)*(y-GW.y)) # distance between node and gateway
     return dist

def get_nearest_gw(x,y):
    nearestGateway = None
    nearestDistance = None
    for gateway in bs:
        distance = get_distance(x,y,gateway)
        if nearestGateway is None:
            nearestGateway = gateway
            nearestDistance = distance
        elif distance < nearestDistance:
            nearestGateway = gateway
            nearestDistance = distance
    return nearestGateway, nearestDistance

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#      
def transmit(env,node):
    global sentPackets_interval, recPackets_interval, lostPackets_interval
    global recPackets
    while True:
        # time before sending anything (include prop delay)
        # simulate the time interval of discrete events happened in a system
        # set_seed(random_seed)
        yield env.timeout(random.expovariate(1.0/float(node.period)))
        
        node.sent = node.sent + nrBS # number of packets sent by the node 
        node.sent_interval += nrBS # number of packets sent by the node during the update interval      
        ParameterConfig.sentPackets_interval += nrBS # number of packets sent by all nodes during the update interval
        global packetSeq
        packetSeq += nrBS # total number of packet of the network
        # print("packetSeq:",packetSeq)

        # generate packets
        if allocation_type == "Local":
            node.Generate_Packet()

        for bs in range(0, nrBS):
            if (node in packetsAtBS[bs]):
                 pass
                # print ("ERROR: packet already in")
            else:
                # adding packet if no collision
                if (checkcollision(node.packet[bs])==1):    
                    node.packet[bs].collided = 1
                else:
                    node.packet[bs].collided = 0
                packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
                # print("node.packet[bs].seqNr:",node.packet[bs].seqNr)
            node.EnergyConsumption += node.packet[bs].tx_energy
            
            ParameterConfig.TotalPacketSize += node.packet[bs].PS
            ParameterConfig.TotalEnergyConsumption += node.packet[bs].tx_energy
            ParameterConfig.TotalPacketAirtime += float(node.packet[bs].rectime / 1000)            
            
        # take first packet time on air   
        yield env.timeout(node.packet[0].rectime)

        # if packet did not collide, add it in list of received packets
        # unless it is already in
        for bs in range(0, nrBS):
            if node.packet[bs].lost:
                ParameterConfig.lostPackets.append(node.packet[bs].seqNr)
                node.lost_interval += 1
                ParameterConfig.lostPackets_interval += 1
            else:
                if node.packet[bs].collided == 0:
                    if (nrNetworks == 1):
                        packetsRecBS[bs].append(node.packet[bs].seqNr)
                        # print("num of packets received by bs",bs,"is",len(packetsRecBS[bs]))
                        node.rec_interval += 1
                        ParameterConfig.recPackets_interval += 1
                    else:
                        # now need to check for right BS
                        if (node.bs.id == bs):
                            packetsRecBS[bs].append(node.packet[bs].seqNr)
                            node.rec_interval += 1
                            ParameterConfig.recPackets_interval += 1
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
                    node.lost_interval += 1
                    ParameterConfig.lostPackets_interval += 1
            # print("num of total received packets",len(ParameterConfig.recPackets))
        

        if allocation_method == "MAB":
            for bs in range(0, nrBS):
                if node.packet[bs].lost == 1 or node.packet[bs].collided == 1:
                    node.agent.reward = -1 # packet loss, negative reward
                else:
                    node.agent.reward = 1 # successully received, positive reward
            # print(node.id)
            # print(node.agent.reward)
            node.agent.cumulative_reward += node.agent.reward
            node.agent.Expected_Reward_Update(node.sf_index, node.bw_index, node.fre_index, node.tp_index)

            # if node.id == 0:
            #     print("Transmission Power",ParameterConfig.Transmission_Power[node.tp_index])

        # if allocation_method == "MAB":
        #     for bs in range(0, nrBS):
        #         if node.packet[bs].lost == "True":
        #             #node.agent.rewards = [-1,-1,-1]
        #             node.agent.rewards = [0,0,-1] # packet lost, negative reward
        #         elif node.packet[bs].collided == 1: 
        #             #node.agent.rewards = [-1,-1,-1]
        #             node.agent.rewards = [-1,-1,0] # packet collided, negative reward
        #         else:
        #             #tp_reward = 5-(5*(float(Transmission_Power[node.tp_index]-8)/10))
        #             tp_reward = 5
        #             #node.agent.rewards = [1,1,1] 
        #             node.agent.rewards = [5,5,tp_reward] # successully received, positive reward
        #     # print("node id:",node.id)
        #     # print(node.agent.reward)
        #     node.agent.cumulative_reward_SF_BW += node.agent.rewards[0]
        #     node.agent.cumulative_reward_Fre += node.agent.rewards[1]
        #     node.agent.cumulative_reward_TP += node.agent.rewards[2]

        #     node.agent.Expected_Reward_Update(node.sf_bw_index, node.fre_index, node.tp_index)

            # if node.id == 0:
            #     print("Q_SF_BW",node.agent.Q_SF_BW)
            #     print("Q_Fre",node.agent.Q_Fre)
            #     print("Q_TP",node.agent.Q_TP)

        
        if allocation_method == "Q-table":
            actions = [node.sf_index,node.bw_index,node.fre_index]
            for bs in range(0, nrBS):
                if node.packet[bs].lost == 1 or node.packet[bs].collided == 1:
                    node.agent.reward = -1 # packet loss, negative reward
                else:
                    node.agent.reward = 5 # successully received, positive reward
            # print(node.id)
            # print(node.agent.reward)
            node.agent.cumulative_reward += node.agent.reward
            if ParameterConfig.Q_table_Config.experience_replay == True:
                node.agent.buffer.push(actions,node.agent.reward)
                node.agent.update_with_experience_replay()
            else:
                 node.agent.update_without_experience_replay(actions)
                
        # print('rec_interval:', node.rec_interval)
        # print('recPackets_interval:', ParameterConfig.recPackets_interval)
        # print('lost_interval:', node.lost_interval)
        # print('lostPackets_interval:', ParameterConfig.lostPackets_interval)

        # complete packet has been received by base station
        # can remove it for next transmission
        for bs in range(0, nrBS):                    
            if (node in packetsAtBS[bs]):
                packetsAtBS[bs].remove(node)
                # reset the packet
                node.packet[bs].collided = 0
                # node.packet[bs].processed = 0