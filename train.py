import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
from MARL.utils import Rollout,setup_seed
import random
def train(nodes):
    # episode_reward_lst = []
    # log = defaultdict(list)
    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(os.getcwd(), "MARL-results")
    folder_path = os.path.join(folder_path,file_name)
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    result_file_name = "traning-result.txt"
    file_path = os.path.join(folder_path, result_file_name)

    setup_seed(MAA2C_Config.random_seed)

    for episode in range(MAA2C_Config.num_episode):
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
        # the settings of the nodes are allocated randomly for the first episode
        ParameterConfig.total_simtime = 0

        if  episode == 0:
            for node in nodes:
                node.agent.action[0] = random.randint(0,5)
        else:
            for node in nodes:
                # update configurations of nodes
                node.agent.action, node.agent.logp_action = node.agent.get_action(torch.as_tensor(node.agent.partial_observation).float().to(MAA2C_Config.device))       
        
        ParameterConfig.total_simtime += action_choose_interval
        # interact with the environment
        env.run(until=ParameterConfig.total_simtime)
        
        # first local observation of agents in the episode
        for node in nodes:
            node.agent.partial_observation = Partial_observation(node)
        # first global observation in the episode
        ParameterConfig.global_observation = Global_observation(nodes)
        # ParameterConfig.global_observation = torch.tensor(ParameterConfig.global_observation).float().to(MAA2C_Config.device)

        interval_count = 0 
        
        while ParameterConfig.total_simtime < simtime: # episode stop condition, conduct each interval
            interval_count += 1
            # intialize the total number of package sent/lost/rec in the interval
            ParameterConfig.sentPackets_interval = 0
            ParameterConfig.recPackets_interval = 0
            ParameterConfig.lostPackets_interval = 0

            for node in nodes:
                node.agent.partial_observation = Partial_observation(node)
                # choose action for each step (update configuration for each node per interval)
                node.agent.action, node.agent.logp_action = node.agent.get_action(torch.as_tensor(node.agent.partial_observation).float().to(MAA2C_Config.device))
                
                # numbers of package lost/received/lost of each node in each interval
                node.sent_interval = 0 
                node.rec_interval = 0 
                node.lost_interval = 0
                                  
            ParameterConfig.total_simtime += action_choose_interval
            # interact with the environment
            env.run(until=ParameterConfig.total_simtime) 

            # print("Obtain reward and observation")
            for node in nodes:
                node.agent.reward = reward_calculation(node)
                node.agent.cumulative_reward += node.agent.reward
                # print("Reward for node",node.id,":",node.agent.reward)
                node.agent.partial_observation = Partial_observation(node) # partial observation of each node
            ParameterConfig.next_global_observation = Global_observation(nodes) # next global observation

            # ParameterConfig.next_global_observation = torch.tensor(ParameterConfig.next_global_observation).float().to(MAA2C_Config.device)
            # print(type(ParameterConfig.global_observation))
            # print("Store experience")
            for node in nodes:
                # print("Reward:",node.agent.reward)
                node.agent.rollout_buffer.put(
                    ParameterConfig.global_observation,
                    node.agent.logp_action,
                    ParameterConfig.next_global_observation,
                    node.agent.reward,
                )

            ParameterConfig.global_observation = ParameterConfig.next_global_observation
            print(f"episode:{episode}, action choose interval:{interval_count}, total sim time:{ParameterConfig.total_simtime}")
        # end time of the simulation of the episode

        # average cumulative reward of the node in this episode
        total_cumulative_reward = 0
        for node in nodes:
            total_cumulative_reward += node.agent.cumulative_reward
        ParameterConfig.average_cumulative_reward.append(total_cumulative_reward/len(nodes))

        end_sim_time = time.time()
        sim_time_per_episode = end_sim_time - start_sim_time 

        # update each agent after one episode
        # start time of the parameter update of the agents
        start_update_time = time.time()
        
        parameter_update(nodes) # update the parameters of the networks of each node

        # end time of the parameter update of the agents
        end_update_time = time.time()
        update_duration = end_update_time - start_update_time

        # if episode % 10 == 0:
        num_rec = len(ParameterConfig.recPackets)
        num_lost = len(ParameterConfig.collidedPackets) + len(ParameterConfig.lostPackets)
        print("num of sent packets in this episode:",)
        print("number of received packets in this episode:",len(ParameterConfig.recPackets))
        print("number of collided packets in this episode:",len(ParameterConfig.collidedPackets))
        print("number of lost packets in this episode",len(ParameterConfig.lostPackets))
        num_sent = num_rec + num_lost
        pdr = float(num_rec/num_sent)*100

        ParameterConfig.PDR.append(pdr)

        with open(file_path, 'a') as file:
            file.write(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}, parameter update time={update_duration:.2f}\n")
        print(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}, actual sim duration={sim_time_per_episode:.2f}, parameter update time={update_duration:.2f}")
    
    # end of training
    
    fig_return = plt.figure()
    x = range(1,MAA2C_Config.num_episode+1)
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

# def eval(args):
#     env = simple_spread_v2.env(N=args.num_agents, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode="human")
#     central_controller = MAC(num_agents=args.num_agents, num_states=args.num_states, num_actions=args.num_actions)

#     agent2policynet = torch.load(os.path.join(args.output_dir, "model.pt"))
#     for agent, state_dict in agent2policynet.items():
#         central_controller.agent2policy[agent].load_state_dict(state_dict)

#     central_controller.eval()

#     episode_reward_lst = []
#     for episode in range(10):
#         episode_reward = 0

#         env.reset()
#         for i, agent in enumerate(env.agent_iter()):
#             state = [env.observe(f"agent_{x}") for x in range(MAA2C_Config.num_agents)]
#             state = np.concatenate(state)

#             action, _ = central_controller.policy(torch.as_tensor(state).float(), agent)
#             env.step(action)

#             if env.agent_selection == "agent_0":
#                 next_state = [env.observe(f"agent_{x}") for x in range(MAA2C_Config.num_agents)]
#                 next_state = np.concatenate(next_state)
#                 reward = env.rewards["agent_0"]
#                 done = env.terminations["agent_0"] or env.truncations["agent_0"]
#                 state = next_state

#                 episode_reward += reward

#                 time.sleep(0.1)

#                 if done is True:
#                     episode_reward_lst.append(episode_reward)
#                     avg_reward = np.mean(episode_reward_lst[-20:])
#                     print(f"episode={episode}, episode reward={episode_reward}, moving reward={avg_reward:.2f}")
#                     break

def reward_calculation(node):
    node_positive_reward = node.rec_interval * MAA2C_Config.receive_reward
    node_negative_reward = node.lost_interval * MAA2C_Config.lost_reward
    # print("packets receive for node ",node.id,"is:",node.rec_interval)
    # print("packets lost for node ",node.id,"is:",node.lost_interval)
    if ((node.rec_interval + node.lost_interval)!=0):
        r_node = float((node_positive_reward + node_negative_reward)/(node.rec_interval + node.lost_interval))
    else:
        r_node = 0
    net_positive_reward = ParameterConfig.recPackets_interval * MAA2C_Config.receive_reward
    net_negative_reward = ParameterConfig.lostPackets_interval * MAA2C_Config.lost_reward
    r_net = float((net_positive_reward + net_negative_reward)/(ParameterConfig.recPackets_interval+ParameterConfig.lostPackets_interval))
    reward = MAA2C_Config.fairness_weight * r_node + (1 - MAA2C_Config.fairness_weight) * r_net
    return reward

def Partial_observation(node):
    partial_observation = []
    partial_observation.append(node.x)
    partial_observation.append(node.y)
    packets = node.packets_interval[-10:]
    for packet in packets:
        if packet.collided == 1 or packet.lost == True:
            partial_observation.append(0)
        else:
            partial_observation.append(1)
    if len(packets) < 12:
        partial_observation = partial_observation + [0] * (12-len(partial_observation))
    partial_observation.append(node.sent_interval)
    partial_observation.append(node.rec_interval)
    partial_observation.append(node.lost_interval)           
    return partial_observation

def Global_observation(nodes):
    Global_Observation = []
    for node in nodes:
        Global_Observation.extend(node.agent.partial_observation)
    return Global_Observation

def parameter_update(nodes):
    for node in nodes:
            # extract the experiemnce in the rollout buffer
            bgo, blogp_a, bngo, br = node.agent.rollout_buffer.tensor()
            # initialize the rollout buffer for next episode
            node.agent.rollout_buffer=Rollout()
            # initialize the cumulative reward of each node for next episode
            node.agent.cumulative_reward = 0

            # print("shape of batch global observation:",bgo.shape)
            # print("shape of batch logp_a:",blogp_a.shape)
            # print("shape of batch reward:",br.shape)
            # print("shape of batch next global observation:",bngo.shape)

            # update the value network
            value_loss, advantage = node.agent.compute_value_loss(bgo, br, bngo)
            node.agent.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            node.agent.value_optimizer.step()

            # update the target network 
            node.agent.soft_update()
            
            # update the actor network
            policy_loss = node.agent.compute_policy_loss(blogp_a, advantage)
            policy_loss.requires_grad_(True)
            node.agent.pi_optimizer.zero_grad()
            policy_loss.backward()
            node.agent.pi_optimizer.step()
    