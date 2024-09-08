import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
from Node import *
from Gateway import *
import random
def MAB_train(nodes):
    # episode_reward_lst = []
    # log = defaultdict(list)
    if ParameterConfig.storage_flag == 1:
        result_folder_path, result_file_path = result_file()

    # set_seed(random_seed)

    for episode in range(MAB_Config.num_episode+1):
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
            node.TotalPacketAirtime = 0 # total packet airtime of the node
            node.Throughput = 0 # throughput of the node
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
        ParameterConfig.ThroughputPerNode = []

        # interact with the environment
        if episode == MAB_Config.num_episode:
            set_seed(random_seed) # reinitialize the random seed
            env.run(until=ParameterConfig.simtime)
        else:
            if episode == 0:
                set_seed(random_seed)
            # set_seed(random_seed) # reinitialize the random seed
            env.run(until=ParameterConfig.traininterveltime)

        if MAB_Config.MAB_Variant == 1:
            MAB_Config.decay_epsilon -=  float(MAB_Config.decay_epsilon / MAB_Config.num_episode)

        # average cumulative rewards of the nodes in this episode
        total_cumulative_reward_SF = 0 
        total_cumulative_reward_BW = 0
        total_cumulative_reward_Fre = 0
        total_cumulative_reward_TP = 0
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
            total_cumulative_reward_SF += node.agent.cumulative_reward_SF
            total_cumulative_reward_BW += node.agent.cumulative_reward_BW
            total_cumulative_reward_Fre += node.agent.cumulative_reward_Fre
            total_cumulative_reward_TP += node.agent.cumulative_reward_TP
            node.agent.cumulative_reward_SF = 0
            node.agent.cumulative_reward_BW = 0
            node.agent.cumulative_reward_Fre = 0
            node.agent.cumulative_reward_TP = 0

            node.PDR = 100*((node.rec_interval)/float(node.sent_interval))
            ParameterConfig.PDRPerNode.append(node.PDR)
            node.EnergyEfficiency = float(8*node.RecPacketSize / node.EnergyConsumption)
            ParameterConfig.EnergyEfficiencyPerNode.append(node.EnergyEfficiency)
            node.Throughput = 8 * float(node.RecPacketSize / node.TotalPacketAirtime)
            ParameterConfig.ThroughputPerNode.append(node.Throughput)
        

        NetThroughput = 8 * float(ParameterConfig.RecPacketSize) / ParameterConfig.TotalPacketAirtime
        ParameterConfig.MaxThroughput = max(ParameterConfig.ThroughputPerNode)
        MinEnergyEfficiency = min(ParameterConfig.EnergyEfficiencyPerNode)
        MaxEnergyEfficiency = max(ParameterConfig.EnergyEfficiencyPerNode)
        JainFairness = Jain_Fairness_Index(nodes)
        NetEnergyEfficiency = float(8*ParameterConfig.RecPacketSize / ParameterConfig.TotalEnergyConsumption)

        num_rec = len(ParameterConfig.recPackets)
        num_lost = len(ParameterConfig.collidedPackets) + len(ParameterConfig.lostPackets)
        pdr = float(num_rec/sumSent)*100 # Network PDR

        if episode != MAB_Config.num_episode:
            MAB_Config.average_cumulative_reward_SF.append(total_cumulative_reward_SF/len(nodes))
            MAB_Config.average_cumulative_reward_BW.append(total_cumulative_reward_BW/len(nodes))
            MAB_Config.average_cumulative_reward_Fre.append(total_cumulative_reward_Fre/len(nodes))
            MAB_Config.average_cumulative_reward_TP.append(total_cumulative_reward_TP/len(nodes))

            MAB_Config.MinPDR.append(min(ParameterConfig.PDRPerNode))
            MAB_Config.MaxPDR.append(max(ParameterConfig.PDRPerNode))
            MAB_Config.MinEnergyEfficiency.append(min(ParameterConfig.EnergyEfficiencyPerNode))
            MAB_Config.MaxEnergyEfficiency.append(max(ParameterConfig.EnergyEfficiencyPerNode))
            MAB_Config.NetworkEnergyEfficiency.append(NetEnergyEfficiency)
            MAB_Config.JainFairness.append(Jain_Fairness_Index(nodes))
            MAB_Config.Network_PDR.append(pdr)
            MAB_Config.Network_Throughput.append(NetThroughput)

        end_sim_time = time.time()
        sim_time_per_episode = end_sim_time - start_sim_time 

        if ParameterConfig.storage_flag == 1:
            with open(result_file_path, 'a') as file:
                # file.write(f"episode={episode}, mum of sent packets={sumSent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}\n")
                file.write(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, MinEnergyEfficiency={MinEnergyEfficiency:.2f}, MaxEnergyEfficiency={MaxEnergyEfficiency:.2f}, Jain's Fairness Index={JainFairness:.2f}, PDR={pdr:.2f}\n")
                # if allocation_method == "MAB":
                #     for node in nodes:
                #         file.write(f"node id={node.id}, Q_SF = {node.agent.Q_SF}, Q_BW = {node.agent.Q_BW}, Q_Fre = {node.agent.Q_Fre}\n")          
            
        # print(f"episode={episode}, mum of sent packets={sumSent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}")
        # print(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, MinEnergyEfficiency={MinEnergyEfficiency:.2f}, MaxEnergyEfficiency={MaxEnergyEfficiency:.2f}, Jain's Fairness Index={JainFairness:.2f}, PDR={pdr:.2f}")
        print(f"episode={episode}, mum of sent packets={sumSent}, num of lost packets=powerloss:{len(ParameterConfig.lostPackets)}+collided:{len(ParameterConfig.collidedPackets)}={num_lost}, Network Throughput={NetThroughput:.2f}, Network EE={NetEnergyEfficiency:.2f}, PDR={pdr:.2f}")


        if episode == MAB_Config.num_episode:
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
            
            '''Test results'''  
            if ParameterConfig.storage_flag == 1:
                config_file = "MAB-Result.txt" # txt results of each episode in training process
                config_file_path = os.path.join(result_folder_path, config_file)
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
                        file.write('Number of training episodes: {}\n\n'.format(MAB_Config.num_episode))

                    file.write('--------Simulation Results--------\n')
                    file.write("Total number of packets sent: {}\n".format(sumSent))
                    file.write("Number of received packets: {}\n".format(num_rec))
                    file.write("Number of collided packets: {}\n".format(len(ParameterConfig.collidedPackets)))
                    file.write("Number of lost packets: {}\n".format(len(ParameterConfig.lostPackets)))
                    file.write("Total payload size: {} bytes\n".format(ParameterConfig.TotalPacketSize))
                    file.write("Received payload size: {} bytes\n".format(ParameterConfig.RecPacketSize))
                    file.write("Total transmission energy consumption: {:.3f} Joule\n".format(ParameterConfig.TotalEnergyConsumption))
                    file.write("Network PDR(Packet Delivery Rate): {:.2f} %\n".format(pdr))
                    file.write("Network throughput: {:.3f} bps\n".format(NetThroughput))
                    file.write("Network Energy efficiency: {:.3f} bits/mJ\n".format(NetEnergyEfficiency))
                    file.write("Minimum PDR: {:.2f} %\n".format(min(ParameterConfig.PDRPerNode)))
                    file.write("Maximum PDR: {:.2f} %\n".format(max(ParameterConfig.PDRPerNode)))
                    file.write("Network Energy efficiency: {:.3f} bits/mJ\n".format(NetEnergyEfficiency))
                    file.write("Minimum Energy efficiency: {:.3f} bits/mJ\n".format(MinEnergyEfficiency))
                    file.write("Jain's fairness index: {:.3f}".format(JainFairness))
                    file.close()
            
    # end of training
    # show results
    if ParameterConfig.storage_flag == 1:
        plot(result_folder_path)

def result_file():
     mab_folder_path = 'Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/MAB' # store the results folder unber the MAN folder
     results_folder_path = os.path.join(mab_folder_path, 'Results') # results folder stores results of different experiments
     experiment_results_folder = datetime.now().strftime("%Y-%m-%d_%H-%M") # create a results folder for each experiment
     experiment_results_folder_path = os.path.join(results_folder_path,experiment_results_folder)
     ParameterConfig.result_folder_path = experiment_results_folder_path
     '''check whether the folder exists, if not, create it'''
     if not os.path.exists(experiment_results_folder_path):
        os.makedirs(experiment_results_folder_path)
     result_file = "MAB-traning-results.txt" # txt results of each episode in training process
     result_file_path = os.path.join(experiment_results_folder_path, result_file)
     return experiment_results_folder_path, result_file_path
    

def plot(result_folder_path):
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

    '''Network Energy Efficiency'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.NetworkEnergyEfficiency, label='NetworkEnergyEfficiency', color='blue')

    plt.title('Network Energy Efficiency changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency/(bits/mJ)')
    plt.legend()

    fig4_name = 'NetworkEnergyEfficiency_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig4_name))

    '''Jain's Faireness Index'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.JainFairness, label='Jain Faireness Index', color='blue')

    plt.title('Jains Faireness Index changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Jains Faireness Index')
    plt.legend()

    fig5_name = 'JainsFairenessIndex_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig5_name))

    '''Average cumulative reward'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.average_cumulative_reward_SF, '-', color='b')

    plt.title('Average cumulative reward of SF changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')

    fig6_name = 'Average_Cumulative_Reward_SF_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig6_name))

    '''Average cumulative reward of BW'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.average_cumulative_reward_BW, '-', color='b')

    plt.title('Average cumulative reward of BW changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')

    fig7_name = 'Average_Cumulative_Reward_BW_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig7_name))

    '''Average cumulative reward of Fre'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.average_cumulative_reward_Fre, '-', color='b')

    plt.title('Average cumulative reward of Fre changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')

    fig8_name = 'average_cumulative_reward_Fre.png'
    plt.savefig(os.path.join(result_folder_path, fig8_name))

    '''Average cumulative reward of TP'''
    plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, MAB_Config.average_cumulative_reward_TP, '-', color='b')

    plt.title('Average cumulative reward of TP changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')

    fig9_name = 'average_cumulative_reward_TP.png'
    plt.savefig(os.path.join(result_folder_path, fig9_name))   

    '''Network PDR'''
    plt.figure()
    plt.plot(x, MAB_Config.Network_PDR, '-', color='b')
    plt.title('Network PDR changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network PDR')

    fig10_name = 'Network_PDR_plot.png'
    plt.savefig(os.path.join(result_folder_path, fig10_name))

    '''Network Throughput'''
    plt.figure()
    plt.plot(x, MAB_Config.Network_Throughput, '-', color='b')
    plt.title('Network throughput changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network Throughput')

    fig11_name = 'Network_Throughput.png'
    plt.savefig(os.path.join(result_folder_path, fig11_name))