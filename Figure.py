import matplotlib.pyplot as plt
import numpy as np
import os

radius_pdr_data = {
    'Random': [70.76, 59.89, 52.72, 48.59, 39.84],
    'Round-Robin': [75.11, 62.51, 55.19, 51.21, 40.83],
    'ADR': [80.41, 75.64, 70.22, 67.31, 59.25],
    'RSLoRa': [66.62, 56.19,  49.33, 45.42, 39.84],
    'DALoRa': [90.91, 89.83, 88.30, 85.81, 80.14]
}

radius_NetEE_data = {
    'Random': [5.790, 4.901, 4.314, 3.976, 3.260],
    'Round-Robin': [6.633, 5.520, 4.874, 4.522, 3.606],
    'ADR': [5.977, 4.389, 3.375, 2.953, 2.047],
    'RSLoRa': [13.808, 8.624, 6.239, 5.276, 2.960],
    'DALoRa': [10.527, 4.950, 2.791, 2.631, 1.620]
}

radius_Throughput_data = {
    'Random': [431.609, 365.315, 321.570, 296.404, 243.020],
    'Round-Robin': [498.439, 414.796, 366.239, 339.808, 270.949],
    'ADR': [869.959, 718.901, 590.566, 527.757, 379.855],
    'RSLoRa': [418.612, 353.071, 309.995, 285.417, 250.312],
    'DALoRa': [573.615, 551.314, 490.888, 461.912, 304.005]
}

radius_NetPer_data = {
    'Random': [1.698, 1.788, 1.84, 1.893, 2.041],
    'Round-Robin': [1.884, 2.17, 2.038, 2.110, 2.223],
    'ADR': [2.318, 2.244, 2.176, 2.360, 2.283],
    'RSLoRa': [2.218, 2.124, 2.09, 2.082, 1.978],
    'DALoRa': [2.422, 2.341, 2.279, 2.374, 2.220]
}

for key in radius_NetPer_data:
    radius_NetPer_data[key] = [float(x / 3) for x in radius_NetPer_data[key]]
# radius_minimum_pdr_data = {
#     'Random': [71.48, 59.92, 54.20, 48.59],
#     'Round-Robin': [67.45, 46.64, 34.90],
#     'ADR': [54.46, 49.82, 54.46],
#     'MAB': [67.42, 78.32, 73.88]
# }


num_pdr_data = {
    'Random': [59.89, 49.61, 39.42, 31.47, 24.97],
    'Round-Robin': [62.51, 53.47, 42.72, 33.28, 26.90],
    'ADR': [75.64, 69.08, 61.25, 52.18, 41.80],
    'RSLoRa': [56.19, 46.01, 36.73, 29.06, 24.51],
    'DALoRa': [89.83, 81.82, 71.23, 57.24, 47.80]
}

num_NetEE_data = {
    'Random': [4.901, 3.970, 3.217, 2.557, 2.022],
    'Round-Robin': [5.520, 4.669, 3.779, 2.938, 2.376],
    'ADR': [4.389, 4.068, 3.658, 3.103, 2.460],
    'RSLoRa': [8.624, 7.226, 5.893, 4.666, 3.087],
    'DALoRa': [4.950, 3.743, 4.060, 3.250, 2.334]
}

num_Throughput_data = {
    'Random': [365.315, 296.999, 236.338, 189.715, 150.463],
    'Round-Robin': [414.796, 350.739, 282.516, 219.541, 177.522],
    'ADR': [718.901, 660.103, 589.760, 501.843, 399.593],
    'RSLoRa': [353.071, 288.648, 230.965, 182.554, 155.613],
    'DALoRa': [551.314, 539.236, 524.298, 415.546, 324.623]
}

num_NetPer_data = {
    'Random': [1.750, 1.659, 1.50, 1.49, 1.548],
    'Round-Robin': [1.922, 2.236, 1.78, 1.664, 1.777],
    'ADR': [2.352, 2.395, 2.470, 2.576, 2.671],
    'RSLoRa': [2.124, 2.000, 1.906, 1.884, 1.900],
    'DALoRa': [2.341, 2.335, 2.579, 2.526, 2.590]
}
for key in num_NetPer_data:
    num_NetPer_data[key] = [float(x / 3) for x in num_NetPer_data[key]]
# num_minimum_pdr_data = {
#     'Random': [59.92, 48.29, 33.47],
#     'Round-Robin': [46.64, 26.67, 5.23],
#     'ADR': [49.82, 48.30, 43.54],
#     'MAB': [78.32, 62.91, 45.03]
# }

# Network tropology radius
radius = [1000, 1500, 2000, 2500, 3000]

# Number of nodes
num_of_nodes = [50, 100, 150, 200, 250]

# 颜色和标签
colors_rgb = [(128,128,128), (243,177,105), (88,159,243), (167,210,186), (249,65,65)]
colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]

makers = ['o', '+', 's', '^', '*']
labels = list(radius_pdr_data.keys())

'''Network PDR'''
plt.figure(figsize=(8, 6))  # 设置图形大小

bar_width = 50  # 柱子宽度
spacing = bar_width / 5  # 柱子间隔
num_algorithms = len(radius_pdr_data)

for i, (algo, pdr_values) in enumerate(radius_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([r + i * (bar_width+spacing) - offset for r in radius], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Network PDR (%)')
# plt.title('Network PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks(range(0, 101, 10))
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

figure_folder_path = "/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/Figures"
if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
fig_name = 'TropologyRadius_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)


'''Network EE'''
plt.figure(figsize=(8, 6))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(radius_NetEE_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([r + i * (bar_width+spacing) - offset for r in radius], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Network Energy Efficiency (bits/mJ)')
# plt.title('Network Energy Efficiency for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks(range(0, 16, 1))
plt.ylim(0, 15) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkEE_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)

'''Network Throughput'''
plt.figure(figsize=(8, 6))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(radius_Throughput_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([r + i * (bar_width+spacing) - offset for r in radius], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Network Throughput (bps)')
# plt.title('Network Throughput for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks(range(0, 950, 100))
plt.ylim(0, 900) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkThropughput_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)


'''Network Performance'''
labels = list(radius_NetPer_data.keys())
data = list(radius_NetPer_data.values())

plt.figure(figsize=(8, 6))  # 设置图形大小

for i, (label, values) in enumerate(radius_NetPer_data.items()):
    plt.plot(radius, values, linestyle='-', marker=makers[i], markersize=5, color=colors[i], label=label,  linewidth=0.8)

plt.xlabel('Topology Radius (m)')
plt.ylabel('Network Performance Index')
# plt.title('Network Performance Index for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks([round(num, 3) for num in list(np.arange(0, 1.1, 0.10))])
plt.ylim(0, 1) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkPerformance_line_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)


# '''Minimum Node PDR'''
# plt.figure(figsize=(10, 6))  # 设置图形大小

# for i, (algo, pdr_values) in enumerate(radius_minimum_pdr_data.items()):
#     offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
#     plt.bar([r + i * bar_width - offset for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo)

# plt.xlabel('Topology Radius (m)')
# plt.ylabel('Minimum Node PDR (%)')
# plt.title('Minimum Node PDR for Different LoRa Parameter Allocation Algorithms')
# plt.xticks(radius)
# plt.yticks(range(0, 101, 5))
# plt.ylim(0, 100) 
# plt.legend()
# plt.show()

# fig_name = 'TropologyRadius_MinimumPDR_bar_chart.png'
# plt.savefig(os.path.join(figure_folder_path, fig_name))



'''Network PDR'''
plt.figure(figsize=(8, 6))  # 设置图形大小

bar_width = 5  # 柱子宽度
spacing = bar_width / 5  # 柱子间隔
num_algorithms = len(num_pdr_data)

for i, (algo, pdr_values) in enumerate(num_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([num + i * (bar_width+spacing) - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Number of Nodes')
plt.ylabel('Network PDR (%)')
# plt.title('Network PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
plt.yticks(range(0, 101, 10))
plt.ylim(0, 100) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)

'''Network EE'''
plt.figure(figsize=(8, 6))  # 设置图形大小
for i, (algo, pdr_values) in enumerate(num_NetEE_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([num + i * (bar_width+spacing) - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Number of Nodes')
plt.ylabel('Network Energy Efficiency (bits/mJ)')
# plt.title('Network Energy Efficiency for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
plt.yticks(range(0, 10, 1))
plt.ylim(0, 9) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkEE_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)


'''Network Throughput'''
plt.figure(figsize=(8, 6))  # 设置图形大小
for i, (algo, pdr_values) in enumerate(num_Throughput_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    plt.bar([num + i * (bar_width+spacing) - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.4, label=algo)

plt.xlabel('Number of Nodes')
plt.ylabel('Network Throughput (bps)')
# plt.title('Network Throughput for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
plt.yticks(range(0, 800, 100))
plt.ylim(0, 750) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)    
plt.show()

fig_name = 'NumberofNodes_NetworkThroughput_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)


'''Network Performance'''
labels = list(num_NetPer_data.keys())
data = list(num_NetPer_data.values())

plt.figure(figsize=(8, 6))  # 设置图形大小

for i, (label, values) in enumerate(num_NetPer_data.items()):
    plt.plot(num_of_nodes, values, linestyle='-', marker=makers[i], markersize=5, color=colors[i], label=label, linewidth=0.8)

plt.xlabel('Number of Nodes')
plt.ylabel('Network Performance Index')
# plt.title('Network Performance Index for Different LoRa Parameter Allocation Algorithms')
plt.xticks(num_of_nodes)
# plt.yticks([round(num,3) for num in list(np.arange(1.400, 2.800, 0.1))])
plt.yticks([round(num,3) for num in list(np.arange(0, 1.1, 0.1))])
# plt.ylim(1.400, 2.700) 
plt.ylim(0, 1) 
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkPerformance_line_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500)

# '''Minimum Node PDR'''
# plt.figure(figsize=(10, 6))  # 设置图形大小

# for i, (algo, pdr_values) in enumerate(num_minimum_pdr_data.items()):
#     offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
#     plt.bar([num + i * bar_width - offset for num in num_of_nodes], pdr_values, width=bar_width, color=colors[i], label=algo)

# plt.xlabel('Number of Nodes (m)')
# plt.ylabel('Minimum Node PDR (%)')
# plt.title('Minimum Node PDR for Different LoRa Parameter Allocation Algorithms')
# plt.xticks(num_of_nodes)
# plt.yticks(range(0, 101, 5))
# plt.ylim(0, 100) 
# plt.legend()
# plt.show()

# fig_name = 'NumberofNodes_MinimumPDR_bar_chart.png'
# plt.savefig(os.path.join(figure_folder_path, fig_name))