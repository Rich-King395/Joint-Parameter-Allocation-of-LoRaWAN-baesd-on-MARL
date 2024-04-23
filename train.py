import numpy as np
import torch
import os
from datetime import datetime
from ParameterConfig import *
import ParameterConfig
from Node import transmit
from MARL.utils import Rollout
import random
def train(nodes):
    # episode_reward_lst = []
    # log = defaultdict(list)
    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(os.getcwd(), "MARL-results")
    folder_path = os.path.join(folder_path,file_name)
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    result_file_name = file_name+"-traning-result.txt"
    file_path = os.path.join(folder_path, result_file_name)
    for episode in range(MAA2C_Config.num_episode):
        '''
          initialize the environment at the beginning of episode
        '''
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
        ParameterConfig.recPackets= [] 
        # list of collided packets
        ParameterConfig.collidedPackets=[] 
        # list of lost packets
        ParameterConfig.lostPackets = [] 
        # the settings of the nodes are allocated randomly
        ParameterConfig.total_simtime = 0
        if  episode == 0:
            for node in nodes:
                node.agent.action[0] = random.randint(0,2)
                node.agent.action[1] = random.randint(0,5)
                node.agent.action[2] = random.randint(0,7)
        else:
            for node in nodes:
                node.agent.action, node.agent.logp_action = node.agent.get_action(torch.as_tensor(node.agent.partial_observation).float().to(MAA2C_Config.device))       
        ParameterConfig.total_simtime += update_interval
        env.run(until=ParameterConfig.total_simtime)
        # first local observation of agents
        for node in nodes:
            node.agent.partial_observation = Partial_observation(node)
        # first global observation
        ParameterConfig.global_observation = Global_observation(nodes)
        ParameterConfig.global_observation = torch.tensor(ParameterConfig.global_observation).float().to(MAA2C_Config.device)
        logp_action_dict = {}
        episode_reward = 0
        # rollout = Rollout()
        interval_count = 0 
        while ParameterConfig.total_simtime < simtime: # episode stop condition, conduct each interval
            interval_count += 1
            ParameterConfig.sentPackets_interval = 0
            ParameterConfig.recPackets_interval = 0
            ParameterConfig.lostPackets_interval = 0
            for node in nodes:
                node.agent.partial_observation = torch.as_tensor(node.agent.partial_observation).float().to(MAA2C_Config.device)
                node.agent.action, node.agent.logp_action = node.agent.get_action(node.agent.partial_observation.float())
                logp_action_dict[node.id] = node.agent.logp_action
                node.sent_interval = 0 
                node.rec_interval = 0 
                node.lost_interval = 0                      
            ParameterConfig.total_simtime += update_interval
            env.run(until=ParameterConfig.total_simtime) # conduct action
            for node in nodes:
                node.agent.reward = reward_calculation(node)
                node.agent.partial_observation = Partial_observation(node)
            ParameterConfig.next_global_observation = Global_observation(nodes)
            ParameterConfig.next_global_observation = torch.tensor(ParameterConfig.next_global_observation).float().to(MAA2C_Config.device)
            for node in nodes:
                value_loss, advantage = node.agent.compute_value_loss(ParameterConfig.global_observation, node.agent.reward, ParameterConfig.next_global_observation)
                node.agent.value_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                node.agent.value_optimizer.step()

                node.agent.soft_update

                policy_loss_bw, policy_loss_sf, policy_loss_fre = node.agent.compute_policy_loss(node.agent.logp_action, advantage)
                torch.autograd.set_detect_anomaly(True)
                shared_policy_loss = 0.3 * policy_loss_bw + 0.4 * policy_loss_sf + 0.3 * policy_loss_fre
                # update the parameters of the first two layers
                node.agent.pi_optimizer.zero_grad()
                shared_policy_loss = shared_policy_loss.clone().detach().requires_grad_(True)
                shared_policy_loss.backward(retain_graph=True)
                node.agent.pi_optimizer.step()

                # update the parameters of the third layer
                policy_loss_bw = policy_loss_bw.clone().detach().requires_grad_(True)
                node.agent.pi_optimizer_output_bw.zero_grad()
                policy_loss_bw.backward()
                node.agent.pi_optimizer_output_bw.step()

                policy_loss_sf = policy_loss_sf.clone().detach().requires_grad_(True)
                node.agent.pi_optimizer_output_sf.zero_grad()
                policy_loss_sf.backward()
                node.agent.pi_optimizer_output_sf.step()

                policy_loss_fre = policy_loss_fre.clone().detach().requires_grad_(True)
                node.agent.pi_optimizer_output_fre.zero_grad()
                policy_loss_fre.backward()
                node.agent.pi_optimizer_output_fre.step()

            print(f"episode:{episode}, update interval:{interval_count}, total sim time:{ParameterConfig.total_simtime}")
            ParameterConfig.global_observation = ParameterConfig.next_global_observation
        
        # if episode % 10 == 0:
        num_rec = len(ParameterConfig.recPackets)
        num_lost = len(ParameterConfig.collidedPackets) + len(ParameterConfig.lostPackets)
        num_sent = num_rec + num_lost
        pdr = float(num_rec/num_sent)*100

        with open(file_path, 'a') as file:
            file.write(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}\n")
        print(f"episode={episode}, mum of sent packets={num_sent}, num of received packets={num_rec}, num of lost packets={num_lost}, PDR={pdr:.2f}")


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
    r_node = float((node_positive_reward + node_negative_reward)/(node.rec_interval + node.lost_interval))
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
    return partial_observation

def Global_observation(nodes):
    Global_Observation = []
    for node in nodes:
        Global_Observation.extend(node.agent.partial_observation)
    return Global_Observation

