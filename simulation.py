import random
import os
from ParameterConfig import *
import ParameterConfig
from Gateway import myBS
from Node import myNode, transmit
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
        self.file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_path = os.path.join(os.getcwd(), "results")
        self.folder_path = os.path.join(self.folder_path,self.file_name)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

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
                if allocation_method not in ["MARL", "MAB", "Q-table"]:
                    # create a transmission process for each node
                    env.process(transmit(env,node)) 
            id += 1

        # store nodes and basestation locations
        if storage_flag == 1:  
            node_path = os.path.join(self.folder_path, self.file_name+"-node.txt")
            with open(node_path, 'w') as nfile:
                for node in nodes:
                    nfile.write('{x} {y} {id}\n'.format(**vars(node)))

            basestation = os.path.join(self.folder_path, self.file_name+"-basestation.txt")
            with open(basestation, 'w') as bfile:
                for basestation in bs:
                    bfile.write('{x} {y} {id}\n'.format(**vars(basestation)))

        # prepare show
        if (graphics == 1):
            plt.xlim([-radius, radius])
            plt.ylim([-radius, radius])
            plt.draw()
            plt.show()
            if storage_flag == 1:  
                fig3_name = 'network_tropology.png'
                plt.savefig(os.path.join(self.folder_path, fig3_name))
        
        if allocation_method=="MAB":
            MAB_train(nodes)
        else:
        # traditional algorithms do not need training stage, start simulation until simtime
            env.run(until=simtime)          

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
            PDRPerNode.append(node.PDR)
            node.EnergyEfficiency = node.RecPacketSize / float(node.EnergyConsumption)
            EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
            
        # der = []
        # data extraction rate = Packet Dilvery Rate PDR
        self.PDRAll = 100*(len(recPackets)/float(self.sumSent))
        self.sumder = 0
        for i in range(0, nrBS):
            self.der.append(100*(len(packetsRecBS[i])/float(self.sent[i])))
            self.sumder = self.sumder + self.der[i]
        self.avgDER = (self.sumder)/nrBS
        
        self.throughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
        self.EnergyEfficiency =  ParameterConfig.RecPacketSize / float(ParameterConfig.TotalEnergyConsumption)
        # fairness measure metrics
        self.MinEnergyEfficiency = min(EnergyEfficiencyPerNode)
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

        print("Network PDR(Packet Delivery Rate): {:.2f} %".format(self.PDRAll))
        print("Minimum PDR: {:.2f} %".format(self.PDRMin))
        print("Maximum PDR: {:.2f} %".format(self.PDRMax))
        print ("Total payload size: {} bytes".format(ParameterConfig.TotalPacketSize))
        print ("Received payload size: {} bytes".format(ParameterConfig.RecPacketSize))
        print ("Total transmission energy consumption: {:.3f} Joule".format(ParameterConfig.TotalEnergyConsumption))
        print ("Network throughput: {:.3f} bps".format(self.throughput))
        print ("Network Energy efficiency: {:.3f} bits/mJ".format(self.EnergyEfficiency))
        print ("Minimum Energy efficiency: {:.3f} bits/mJ".format(self.MinEnergyEfficiency))
        print ("Jain's fairness index: {:.3e}".format(self.JainFairness))

        # this can be done to keep graphics visible
        if (graphics == 1):
            input('Press Enter to continue ...')
    
    def simulation_record(self):
        result_file_name = self.file_name+"-result.txt"   
        file_path = os.path.join(self.folder_path, result_file_name)
        with open(file_path, 'w') as file:
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
            for i in range(0, nrBS):
                file.write("send to BS[{}".format(i))
                file.write("]: {}\n".format(self.sent[i])) # number of packets sent to each BS
            for i in range(0,nrBS):
                file.write("packets at BS {}".format(i))
                file.write(": {}\n".format(len(packetsRecBS[i]))) # received packets of each BS
            file.write("overall received at right BS: {}\n".format(self.sum))
            for i in range(0, nrBS):
                file.write("DER BS[".format(i))
                file.write("]: {:.2f}%\n".format(self.der[i]))    
            file.write("avg DER: {:.2f}%\n".format(self.avgDER))
            file.write("DER with 1 network: {:.2f}%\n".format(self.PDRAll))
            file.write("Total payload size: {} bytes\n".format(ParameterConfig.TotalPacketSize))
            file.write("Received payload size: {} bytes\n".format(ParameterConfig.RecPacketSize))
            file.write("Total transmission energy consumption: {:.3f} Joule\n".format(ParameterConfig.TotalEnergyConsumption))
            file.write("Network throughput: {:.3f} bps\n".format(self.throughput))
            file.write("Network Energy efficiency: {:.3f} bit/Joule\n".format(self.EnergyEfficiency))
            file.write("Minimum Energy efficiency: {:.3f} bits/mJ".format(self.MinEnergyEfficiency))

