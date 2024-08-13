import numpy as np
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
import random
def MAB_train(nodes):
    # episode_reward_lst = []
    # log = defaultdict(list)
    if ParameterConfig.storage_flag == 1:
        result_folder_path, result_file_path = result_file()
        Config_Record(result_folder_path)

    # set_seed(random_seed)

    for episode in range(MAB_Config.num_episode):
        '''
          initialize the environment at the beginning of episode
        '''
        # start time of simulation of this episode
        start_sim_time = time.time()

        # initialize simulation environment current time for each episode
        env = simpy.Environment()
        for node in nodes:
            node.sent_interval = 0 # number of packets sent by the node during update interval
            node.rec_interval = 0 # number of sussccessfully-seceived packets sent by the node during update interval
            node.lost_interval = 0 # number of lost packets sent by the node during update interval

            node.PDR = 0 # packet delivery ratio of the node
            node.RecPacketSize = 0 # size of packets received by the node
            node.EnergyConsumption = 0 # energy consumption of the node
            node.EnergyEfficiency = 0 # energy efficiency of the node
            # create a transmission process for each node
            env.process(transmit(env,node))
        
        # Packets sent to each GW
        ParameterConfig.packetsAtBS = [] 
        # Packets received by each GW
        ParameterConfig.packetsRecBS = [] 
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

        # number of sent packets during update interval 
        ParameterConfig.sentPackets_interval = 0
        # number of received packets during update interval 
        ParameterConfig.recPackets_interval = 0
        # number of not received packets during update interval
        ParameterConfig.lostPackets_interval = 0
       
        ParameterConfig.PDRPerNode = []
        ParameterConfig.EnergyEfficiencyPerNode = []

        # interact with the environment
        env.run(until=ParameterConfig.simtime)

        if MAB_Config.MAB_Variant == 1:
            MAB_Config.decay_epsilon -=  float(MAB_Config.decay_epsilon / MAB_Config.num_episode)

        # average cumulative reward of the node in this episode
        total_cumulative_reward = 0
        sumSent = 0 # total number of packets sent in this episode
        sumPowerLost = 0 # total number of packets lost in this episode
        sumCollided = 0 # total number of packets collided in this episode
        for i in range(0,nrNodes*nrBS):
            sumSent = sumSent + nodes[i].sent
            sumPowerLost = sumPowerLost + nodes[i].powerlost
            sumCollided = sumCollided + nodes[i].collided
        for node in nodes:
            node.sent = 0
            node.collided = 0
            node.powerlost = 0
            total_cumulative_reward += node.agent.cumulative_reward
            node.agent.cumulative_reward = 0

            node.PDR = 100*((node.rec_interval)/float(node.sent_interval))
            ParameterConfig.PDRPerNode.append(node.PDR)
            node.EnergyEfficiency = node.RecPacketSize / float(node.EnergyConsumption)
            ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)

        MinEnergyEfficiency = min(ParameterConfig.EnergyEfficiencyPerNode)
        MaxEnergyEfficiency = max(ParameterConfig.EnergyEfficiencyPerNode)
        JainFairness = Jain_Fairness_Index(nodes)

        MAB_Config.average_cumulative_reward.append(total_cumulative_reward/len(nodes))
        MAB_Config.MinPDR.append(min(ParameterConfig.PDRPerNode))
        MAB_Config.MaxPDR.append(max(ParameterConfig.PDRPerNode))
        MAB_Config.MinEnergyEfficiency.append(min(ParameterConfig.EnergyEfficiencyPerNode))
        MAB_Config.MaxEnergyEfficiency.append(max(ParameterConfig.EnergyEfficiencyPerNode))
        MAB_Config.JainFairness.append(Jain_Fairness_Index(nodes))

        end_sim_time = time.time()
        sim_time_per_episode = end_sim_time - start_sim_time 

        num_rec = len(ParameterConfig.recPackets)
        num_lost = len(ParameterConfig.collidedPackets) + len(ParameterConfig.lostPackets)
        pdr = float(num_rec/sumSent)*100 # Network PDR

        MAB_Config.Network_PDR.append(pdr)

        if ParameterConfig.storage_flag == 1:
            with open(result_file_path, 'a') as file:
                # file.write(f"episode={episode}, mum of sent packets={sumSent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}\n")
                file.write(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, MinEnergyEfficiency={MinEnergyEfficiency:.2f}, MaxEnergyEfficiency={MaxEnergyEfficiency:.2f}, Jain's Fairness Index={JainFairness:.2f}, PDR={pdr:.2f}\n")
                # if allocation_method == "MAB":
                #     for node in nodes:
                #         file.write(f"node id={node.id}, Q_SF = {node.agent.Q_SF}, Q_BW = {node.agent.Q_BW}, Q_Fre = {node.agent.Q_Fre}\n")          
            
        # print(f"episode={episode}, mum of sent packets={sumSent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}")
        print(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, MinEnergyEfficiency={MinEnergyEfficiency:.2f}, MaxEnergyEfficiency={MaxEnergyEfficiency:.2f}, Jain's Fairness Index={JainFairness:.2f}, PDR={pdr:.2f}")

    
    # end of training
    # show results
    if ParameterConfig.storage_flag == 1:
        '''Max PDR & Min PDR'''
        plt.figure()
        x = range(1,MAB_Config.num_episode+1)
        plt.plot(x, MAB_Config.MaxPDR, label='MaxPDR', color='blue')
        plt.plot(x, MAB_Config.MinPDR, label='MinPDR', color='red')

        plt.title('Max PDR and Min PDR changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('PDR')
        plt.legend()

        fig1_name = 'MaxMinPDR_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig1_name))

        '''Min Energy Efficiency'''
        plt.figure()
        x = range(1,MAB_Config.num_episode+1)
        plt.plot(x, MAB_Config.MinEnergyEfficiency, label='MinEnergyEfficiency', color='blue')

        plt.title('Minimum Energy Efficiency changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('Energy Efficiency/(bits/mJ)')
        plt.legend()

        fig2_name = 'MinEnergyEfficiency_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig2_name))

        '''Max Energy Efficiency'''
        plt.figure()
        x = range(1,MAB_Config.num_episode+1)
        plt.plot(x, MAB_Config.MaxEnergyEfficiency, label='MaxEnergyEfficiency', color='blue')

        plt.title('Maximum Energy Efficiency changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('Energy Efficiency/(bits/mJ)')
        plt.legend()

        fig3_name = 'MaxEnergyEfficiency_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig3_name))

        '''Jain's Faireness Index'''
        plt.figure()
        x = range(1,MAB_Config.num_episode+1)
        plt.plot(x, MAB_Config.JainFairness, label='Jain Faireness Index', color='blue')

        plt.title('Jains Faireness Index changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('Jains Faireness Index')
        plt.legend()

        fig4_name = 'JainsFairenessIndex_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig4_name))

        '''Average cumulative reward'''
        plt.figure()
        x = range(1,MAB_Config.num_episode+1)
        plt.plot(x, MAB_Config.average_cumulative_reward, '-', color='b')

        plt.title('Average cumulative reward changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Cumulative Reward')

        fig5_name = 'Average_Cumulative_Reward_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig5_name))

        '''Network PDR'''
        plt.figure()
        plt.plot(x, MAB_Config.Network_PDR, '-', color='b')
        plt.title('Network PDR changes with increasing episode')
        plt.xlabel('Episode')
        plt.ylabel('Network PDR')

        fig6_name = 'Network_PDR_plot.png'
        plt.savefig(os.path.join(result_folder_path, fig6_name))

def result_file():
     mab_folder_path = 'Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MAB' # store the results folder unber the MAN folder
     results_folder_path = os.path.join(mab_folder_path, 'Results') # results folder stores results of different experiments
     experiment_results_folder = datetime.now().strftime("%Y-%m-%d_%H-%M") # create a results folder for each experiment
     experiment_results_folder_path = os.path.join(results_folder_path,experiment_results_folder)
     '''check whether the folder exists, if not, create it'''
     if not os.path.exists(experiment_results_folder_path):
        os.makedirs(experiment_results_folder_path)
     result_file = "MAB-traning-results.txt" # txt results of each episode in training process
     result_file_path = os.path.join(experiment_results_folder_path, result_file)
     return experiment_results_folder_path, result_file_path

def Config_Record(experiment_results_folder_path):
    config_file = "MAB-config.txt" # txt results of each episode in training process
    config_file_path = os.path.join(experiment_results_folder_path, config_file)
    with open(config_file_path, 'w') as file:
        file.write('--------Network Parameter Setting--------\n')
        file.write('Nodes per base station: {}\n'.format(ParameterConfig.nrNodes))
        file.write('Packet generation interval: {} ms\n'.format(ParameterConfig.avgSendTime))
        file.write('LoRa parameters allocation type: {}\n'.format(ParameterConfig.allocation_type))
        file.write('LoRa parameters allocation method: {}\n'.format(ParameterConfig.allocation_method))
        file.write('Simulation duration: {} h\n'.format(round(ParameterConfig.simtime/3600000, 2)))
        file.write('Number of gateways: {}\n'.format(ParameterConfig.nrBS))
        if full_collision == 1:
            file.write('Collision check mode: Full Collision Check\n')
        else:
            file.write('Collision check mode: Simple Collision Check\n')
        if directionality == 1:
            file.write('Antenna type: Directional antenna\n')
        else:
            file.write('Antenna type: Omnidirectional antenna\n')
        file.write('Number of networks: {}\n'.format(ParameterConfig.nrNetworks))
        file.write('Network topology radius: {} m\n'.format(ParameterConfig.radius))
        file.write('Packet payload size: {}\n\n'.format(ParameterConfig.PayloadSize))
        file.write('--------MAB Configuration--------\n')
        if MAB_Config.MAB_Variant == 0:
            file.write('MAB type: Epsilon Greedy\n')
            file.write('Epsilon: {}\n'.format(MAB_Config.epsilon))
            file.write('Number of training episodes: {}\n'.format(MAB_Config.num_episode))
        elif MAB_Config.MAB_Variant == 1:
            file.write('MAB type: Decaying-Epsilon Greedy\n')
        elif MAB_Config.MAB_Variant == 2:
            file.write('MAB type: UCB(Upper Confidence Bound)\n')
            file.write('Alpha: {}\n'.format(MAB_Config.coef))
            file.write('Number of training episodes: {}\n'.format(MAB_Config.num_episode))
