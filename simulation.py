import random
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from ParameterConfig import *
import ParameterConfig
from Gateway import myBS, graphics_gateway
from Node import myNode, transmit, graphics_node
from datetime import datetime
from MAB.train import MAB_train
class Simulation:
    def __init__(self):
        self.sum = 0
        self.sumSent = 0
        self.sent = [] 
        self.der = []
        self.simstarttime = 0
        self.simendtime = 0
        self.avgDER = 0
        self.PDRAll = 0

        self.PDRMin = 0
        self.PDRMax = 0
        self.throughput = 0
        self.EnergyEfficiency = 0 # Network Energy efficiency (bits/mJoule)

        self.MinEnergyEfficiency = 0 # Minimum energy efficiency among the nodes (bits/mJoule)
        self.JainFairness = 0 # Jain's fairness index
        ParameterConfig.result_folder_path, self.result_file = result_file()

    def run(self):
        # generate BS
        for i in range(0,nrBS):
            b = myBS(i)
            bs.append(b) # append the BS to the list
            # append new list for each BS
            packetsAtBS.append([]) 
            packetsRecBS.append([])

        set_seed(random_seed)
        # generate node
        id = 0
        while len(nodes) < nrNodes*nrBS:
            # myNode takes period (in ms), base station id packetlen (in Bytes)
            # 1000000 = 16 min
            x = random.randint(-radius, radius)
            y = random.randint(-radius, radius)
            # make sure the nodes are inside the circle
            if (x ** 2 + y ** 2) > (radius ** 2):
                continue
            for j in range(0,nrBS):
                # create nrNodes for each base station
                # For different BS, same node has different id
                node = myNode(id*nrBS+j,x,y,avgSendTime,bs[j])
                nodes.append(node)
                
                # when we add directionality, we update the RSSI here
                if (directionality == 1):
                    node.updateRSSI()
                if allocation_method not in ["MARL", "DALoRa", "Q-table"]:
                    # create a transmission process for each node
                    env.process(transmit(env,node)) 
            id += 1
        
        if allocation_method=="DALoRa":
            set_seed(random_seed)
            MAB_train(nodes)
        else:
        # traditional algorithms do not need training stage, start simulation until simtime
            set_seed(random_seed)
            env.run(until=simtime)

        # store nodes and basestation locations
        if storage_flag == 1:  
            node_path = os.path.join(ParameterConfig.result_folder_path, "node.txt")
            with open(node_path, 'w') as nfile:
                for node in nodes:
                    nfile.write('{x} {y} {id}\n'.format(**vars(node)))

            basestation = os.path.join(ParameterConfig.result_folder_path, "basestation.txt")
            with open(basestation, 'w') as bfile:
                for basestation in bs:
                    bfile.write('{x} {y} {id}\n'.format(**vars(basestation)))

        # prepare show
        # if (graphics == 1):
        #     plt.figure(figsize=(6, 6))
        #     ax = plt.gcf().gca()
        #     for GW in bs:
        #         graphics_gateway(GW,ax)
        #     for node in nodes:
        #         graphics_node(node,ax)
        #     plt.xlim([-radius, radius])
        #     plt.ylim([-radius, radius])
        #     # plt.draw()
        #     # plt.show()
        #     if storage_flag == 1:  
        #         fig_name = 'network_tropology.png'
        #         plt.savefig(os.path.join(ParameterConfig.result_folder_path, fig_name))          

    def results_calculation(self):
        for i in range(0,nrBS):
            self.sum = self.sum + len(packetsRecBS[i]) # calculate total received packets
        for i in range(0,nrBS):
            self.sent.append(0)
        for i in range(0,nrNodes*nrBS):
            self.sumSent = self.sumSent + nodes[i].sent
            #print ("id for node ", nodes[i].id, "BS:", nodes[i].bs.id, " sent: ", nodes[i].sent)
            self.sent[nodes[i].bs.id] = self.sent[nodes[i].bs.id] + nodes[i].sent
        for node in nodes:
            node.PDR = 100*((node.rec_interval)/float(node.sent_interval))
            node.EnergyEfficiency = node.RecPacketSize / float(node.EnergyConsumption)
            node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
            PDRPerNode.append(node.PDR)
            EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
            ThroughputPerNode.append(node.Throughput)
            
        # der = []
        # data extraction rate = Packet Dilvery Rate PDR
        self.PDRAll = 100*(len(recPackets)/float(self.sumSent))
        self.sumder = 0
        for i in range(0, nrBS):
            self.der.append(100*(len(packetsRecBS[i])/float(self.sent[i])))
            self.sumder = self.sumder + self.der[i]
        self.avgDER = (self.sumder)/nrBS
        
        self.throughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
        self.EnergyEfficiency =  8 * float(ParameterConfig.RecPacketSize) / float(ParameterConfig.TotalEnergyConsumption)
        # fairness measure metrics
        self.MinEnergyEfficiency = min(EnergyEfficiencyPerNode)
        ParameterConfig.MaxThroughput = max(ThroughputPerNode)
        self.PDRMin = min(PDRPerNode)
        self.PDRMax = max(PDRPerNode)
        self.JainFairness = Jain_Fairness_Index(nodes)



    def results_show(self):
        # print stats and save into file
        print ("Total number of packets sent: ", self.sumSent)
        print ("Number of received packets (independent of right base station)", len(recPackets))
        print ("Number of collided packets", len(collidedPackets))
        print ("Number of lost packets (not correct)", len(lostPackets))
        
        # for i in range(0, nrBS):
        #     print ("send to BS[",i,"]:", self.sent[i]) # number of packets sent to each BS
        # global packetSeq
        # print ("sent packets: ", packetSeq) # total sent packets of nodes
        # for i in range(0,nrBS):
        #     print ("packets at BS",i, ":", len(packetsRecBS[i])) # received packets of each BS
        # print ("overall received at right BS: ", self.sum)

        # for i in range(0, nrBS):
        #     print ("DER BS[",i,"]: {:.2f}".format(self.der[i]))    
        # print ("avg DER: {:.2f}".format(self.avgDER))
        # print ("DER with 1 network:{:.2f}".format(self.PDRAll))

        print ("Total payload size: {} bytes".format(ParameterConfig.TotalPacketSize))
        print ("Received payload size: {} bytes".format(ParameterConfig.RecPacketSize))
        print ("Total transmission energy consumption: {:.3f} Joule".format(ParameterConfig.TotalEnergyConsumption))
        print ("Network throughput: {:.3f} bps".format(self.throughput))
        print("Network PDR(Packet Delivery Rate): {:.2f} %".format(self.PDRAll))
        print("Minimum PDR: {:.2f} %".format(self.PDRMin))
        print("Maximum PDR: {:.2f} %".format(self.PDRMax))
        print ("Network Energy efficiency: {:.3f} bits/mJ".format(self.EnergyEfficiency))
        print ("Minimum Energy efficiency: {:.3f} bits/mJ".format(self.MinEnergyEfficiency))
        print ("Jain's fairness index: {:.3e}".format(self.JainFairness))

        # this can be done to keep graphics visible
        if (graphics == 1):
            input('Press Enter to continue ...')
    
    def simulation_record(self):
        '''Heatmap'''
         # prepare show
        if (graphics == 1):
            plt.figure(figsize=(6, 6))
            ax = plt.gcf().gca()
            for GW in bs:
                graphics_gateway(GW,ax)
            for node in nodes:
                graphics_node(node,ax)

            # 保证生成的图像是正方形
            ax.set_aspect('equal')

            plt.xlim([-radius, radius])
            plt.ylim([-radius, radius])
            
            plt.tick_params(axis='x', direction='in')  
            plt.tick_params(axis='y', direction='in')  
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            # 创建虚拟 ScalarMappable 对象来生成颜色条
            sm = ScalarMappable(cmap=cm.plasma)
            sm.set_array([])  # 设置一个空数组，因为颜色映射实际上不需要数据

            # 添加颜色条到图像
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Throughput')  # 设置颜色条的标签

            if storage_flag == 1:  
                fig_name = 'network_tropology.png'
                plt.savefig(os.path.join(ParameterConfig.result_folder_path, fig_name), dpi=800)   

        '''Simulation Results'''
        with open(self.result_file, 'w') as file:
            file.write('Simulation start at {}'.format(self.simstarttime))
            file.write(' and end at {}\n'.format(self.simendtime))
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
            file.write("Total number of packets sent: {}\n".format(self.sumSent))
            file.write("Number of received packets: {}\n".format(len(recPackets)))
            file.write("Number of collided packets: {}\n".format(len(collidedPackets)))
            file.write("Number of lost packets: {}\n".format(len(lostPackets)))

            # for i in range(0, nrBS):
            #     file.write("send to BS[{}".format(i))
            #     file.write("]: {}\n".format(self.sent[i])) # number of packets sent to each BS
            # for i in range(0,nrBS):
            #     file.write("packets at BS {}".format(i))
            #     file.write(": {}\n".format(len(packetsRecBS[i]))) # received packets of each BS
            # file.write("overall received at right BS: {}\n".format(self.sum))
            # for i in range(0, nrBS):
            #     file.write("DER BS[".format(i))
            #     file.write("]: {:.2f}%\n".format(self.der[i]))  

            # file.write("avg DER: {:.2f}%\n".format(self.avgDER))
            # file.write("DER with 1 network: {:.2f}%\n".format(self.PDRAll))

            file.write("Total payload size: {} bytes\n".format(ParameterConfig.TotalPacketSize))
            file.write("Received payload size: {} bytes\n".format(ParameterConfig.RecPacketSize))
            file.write("Total transmission energy consumption: {:.3f} Joule\n".format(ParameterConfig.TotalEnergyConsumption))
            file.write("Network throughput: {:.3f} bps\n".format(self.throughput))
            file.write("Network PDR(Packet Delivery Rate): {:.2f} %\n".format(self.PDRAll))
            file.write("Minimum PDR: {:.2f} %\n".format(self.PDRMin))
            file.write("Maximum PDR: {:.2f} %\n".format(self.PDRMax))
            file.write("Network Energy efficiency: {:.3f} bits/mJ\n".format(self.EnergyEfficiency))
            file.write("Minimum Energy efficiency: {:.3f} bits/mJ\n".format(self.MinEnergyEfficiency))
            file.write("Jain's fairness index: {:.3f}".format(self.JainFairness))

def result_file():
     baseline_folder_path = '/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL' 
     results_folder_path = os.path.join(baseline_folder_path, 'Baseline_Results') # results folder stores results of different experiments
     experiment_results_folder = datetime.now().strftime("%Y-%m-%d_%H-%M") # create a results folder for each experiment
     experiment_results_folder_path = os.path.join(results_folder_path,experiment_results_folder)
     '''check whether the folder exists, if not, create it'''
     if not os.path.exists(experiment_results_folder_path):
        os.makedirs(experiment_results_folder_path)
     result_file = allocation_method+"-Result.txt" # txt results of each episode in training process
     result_file_path = os.path.join(experiment_results_folder_path, result_file)
     return experiment_results_folder_path, result_file_path

