import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
from MAA2C.utils import setup_seed
import random
def Q_table_train(nodes):
    # episode_reward_lst = []
    # log = defaultdict(list)
    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(os.getcwd(), "Q-table-results")
    folder_path = os.path.join(folder_path,file_name)
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    result_file_name = "Q-table-traning-result.txt"
    file_path = os.path.join(folder_path, result_file_name)

    setup_seed(Q_table_Config.random_seed)

    for episode in range(MAB_Config.num_episode):
        '''
          initialize the environment at the beginning of episode
        '''
        # start time of simulation of this episode
        start_sim_time = time.time()

        # initialize simulation environment current time for each episode
        env = simpy.Environment()
        for node in nodes:
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
        
        # interact with the environment
        env.run(until=ParameterConfig.simtime)

        # average cumulative reward of the node in this episode
        total_cumulative_reward = 0
        num_sent = 0
        for node in nodes:
            num_sent += node.sent
            node.sent = 0
            total_cumulative_reward += node.agent.cumulative_reward
            node.agent.cumulative_reward = 0
        ParameterConfig.average_cumulative_reward.append(total_cumulative_reward/len(nodes))

        end_sim_time = time.time()
        sim_time_per_episode = end_sim_time - start_sim_time 

        # update each agent after one episode
        # start time of the parameter update of the agents
        start_update_time = time.time()

        # end time of the parameter update of the agents
        end_update_time = time.time()
        update_duration = end_update_time - start_update_time

        # if episode % 10 == 0:
        num_rec = len(ParameterConfig.recPackets)
        num_lost = len(ParameterConfig.collidedPackets) + len(ParameterConfig.lostPackets)
        # print("num of sent packets in this episode:",)
        # print("number of received packets in this episode:",len(ParameterConfig.recPackets))
        # print("number of collided packets in this episode:",len(ParameterConfig.collidedPackets))
        # print("number of lost packets in this episode",len(ParameterConfig.lostPackets))
        # num_sent = num_rec + num_lost
        pdr = float(num_rec/num_sent)*100

        ParameterConfig.PDR.append(pdr)

        with open(file_path, 'a') as file:
            file.write(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}, parameter update time={update_duration:.2f}\n")
            # if allocation_method == "MAB":
            #     for node in nodes:
            #         file.write(f"node id={node.id}, Q_SF = {node.agent.Q_SF}, Q_BW = {node.agent.Q_BW}, Q_Fre = {node.agent.Q_Fre}\n")          
        print(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}, parameter update time={update_duration:.2f}")

    
    # end of training
    
    fig_return = plt.figure()
    x = range(1,MAB_Config.num_episode+1)
    plt.plot(x, ParameterConfig.average_cumulative_reward, '-', color='b')

    plt.title('Average cumulative reward changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')

    fig1_name = 'return_plot.png'
    plt.savefig(os.path.join(folder_path, fig1_name))

    fig_pdr = plt.figure()
    plt.plot(x, ParameterConfig.PDR, '-', color='b')

    plt.title('Packet delivery rate changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Packet delivery rate')

    fig2_name = 'PDR_plot.png'
    plt.savefig(os.path.join(folder_path, fig2_name))

    