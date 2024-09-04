import matplotlib.pyplot as plt
import os

radius_pdr_data = {
    'Random': [80.68, 74.60, 68.85],
    'Round-Robin': [87.15, 80.07, 73.86],
    'ADR': [62.91, 59.73, 62.91],
    'MAB': [90.38, 88.68, 86.65]
}

radius_minimum_pdr_data = {
    'Random': [71.48, 59.92, 54.20],
    'Round-Robin': [67.45, 46.64, 34.90],
    'ADR': [54.46, 49.82, 54.46],
    'MAB': [67.42, 78.32, 73.88]
}

num_pdr_data = {
    'Random': [74.60, 61.24, 46.00],
    'Round-Robin': [80.07, 65.66, 51.02],
    'ADR': [59.73, 56.74, 50.47],
    'MAB': [88.68, 79.08, 66.79]
}

num_minimum_pdr_data = {
    'Random': [59.92, 48.29, 33.47],
    'Round-Robin': [46.64, 26.67, 5.23],
    'ADR': [49.82, 48.30, 43.54],
    'MAB': [78.32, 62.91, 45.03]
}

# Network tropology radius
radius = [1000, 1500, 2000]

# Number of nodes
num_of_nodes = [50,100,150]

# 颜色和标签
colors = ['b', 'g', 'r', 'c']
labels = list(radius_pdr_data.keys())

'''Network PDR'''
plt.figure(figsize=(10, 6))  # 设置图形大小

bar_width = 50  # 柱子宽度
num_algorithms = len(radius_pdr_data)

for i, (algo, pdr_values) in enumerate(radius_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([r + i * bar_width - offset for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Network PDR (%)')
plt.title('Network PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks(range(0, 101, 5))
plt.ylim(0, 100) 
plt.legend()
plt.show()

figure_folder_path = "/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/Figures"
if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
fig_name = 'TropologyRadius_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name))

'''Minimum Node PDR'''
plt.figure(figsize=(10, 6))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(radius_minimum_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([r + i * bar_width - offset for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Minimum Node PDR (%)')
plt.title('Minimum Node PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks(range(0, 101, 5))
plt.ylim(0, 100) 
plt.legend()
plt.show()

fig_name = 'TropologyRadius_MinimumPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name))



'''Network PDR'''
plt.figure(figsize=(10, 6))  # 设置图形大小

bar_width = 5  # 柱子宽度
num_algorithms = len(num_pdr_data)

for i, (algo, pdr_values) in enumerate(num_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([num + i * bar_width - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], label=algo)

plt.xlabel('Number of Nodes')
plt.ylabel('Network PDR (%)')
plt.title('Network PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
plt.yticks(range(0, 101, 5))
plt.ylim(0, 100) 
plt.legend()
plt.show()

fig_name = 'NumberofNodes_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name))

'''Minimum Node PDR'''
plt.figure(figsize=(10, 6))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(num_minimum_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([num + i * bar_width - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], label=algo)

plt.xlabel('Number of Nodes (m)')
plt.ylabel('Minimum Node PDR (%)')
plt.title('Minimum Node PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
plt.yticks(range(0, 101, 5))
plt.ylim(0, 100) 
plt.legend()
plt.show()

fig_name = 'NumberofNodes_MinimumPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name))