import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os


def Net_Per_Index_Calculation(pdr_data, EE_data, Throughput_data):
    alpha = 1/3
    beta = 1/3
    gamma = 1/3
    weights = [alpha, beta, gamma]
    keys = list(pdr_data.keys())
    pdr_list = []
    pdr_max_values=[]
    EE_list = []
    EE_max_values=[]
    Throughput_list = []
    Throughput_max_values=[]
    pdr_exp = pdr_data
    EE_exp = EE_data
    Throughput_exp = Throughput_data
    for i in range(len(pdr_data[keys[0]])):
        pdr_list.append([pdr_data[key][i] for key in keys])
        EE_list.append([EE_data[key][i] for key in keys])
        Throughput_list.append([Throughput_data[key][i] for key in keys])
        pdr_max_value = max(pdr_list[i])
        pdr_max_values.append(pdr_max_value)
        EE_max_value = max(EE_list[i])
        EE_max_values.append(EE_max_value)
        Throughput_max_value = max(Throughput_list[i])
        Throughput_max_values.append(Throughput_max_value)
        for key in keys:
            pdr_exp[key][i] = float(pdr_data[key][i] / pdr_max_values[i])
            EE_exp[key][i] = float(EE_data[key][i] / EE_max_values[i])
            Throughput_exp[key][i] = float(Throughput_data[key][i] / Throughput_max_values[i])
    Net_Per_Index = {}
    for key in keys:
        weighted_values = [sum(w * x for w, x in zip(weights, data)) for data in zip(pdr_exp[key], EE_exp[key], Throughput_exp[key])]
        Net_Per_Index[key] = weighted_values
    return Net_Per_Index

# radius_node_50_pdr_data = {
#     'Random': [70.76, 59.89, 52.72, 48.59, 39.84],
#     'Round-Robin': [75.11, 62.51, 55.19, 51.21, 40.83],
#     'ADR': [80.41, 75.64, 70.22, 67.31, 59.25],
#     'RSLoRa': [66.62, 56.19,  49.33, 45.42, 39.84],
#     'DALoRa-Balance': [90.91, 89.83, 88.30, 85.81, 80.14],
#     'DALoRa-PDR': [95.30, 92.14, 89.46, 86.70, 79.66],
#     'DALoRa-EE': [84.14, 80.89, 78.73, 79.07, 74.86],
#     'DALoRa-Throughput': [89.91, 83.68, 74.99, 77.65, 72.86]
    
# }

# radius_node_50_NetEE_data = {
#     'Random': [46.32, 39.208, 34.512, 31.808, 26.08],
#     'Round-Robin': [53.064, 44.16, 38.992, 36.176, 28.848],
#     'ADR': [47.816, 35.112, 27.0, 23.624, 16.376],
#     'RSLoRa': [110.464, 68.992, 49.912, 42.208, 23.68],
#     'DALoRa-Balance': [84.216, 39.6, 22.328, 21.048, 12.96],
#     'DALoRa-PDR': [25.673, 23.469, 21.318, 17.253, 13.450],
#     'DALoRa-EE': [125.187, 50.790, 37.688, 23.146, 14.760],
#     'DALoRa-Throughput': [36.688, 26.077, 17.466, 8.863, 8.064]
    
# }

# radius_node_50_Throughput_data = {
#     'Random': [431.609, 365.315, 321.570, 296.404, 243.020],
#     'Round-Robin': [498.439, 414.796, 366.239, 339.808, 270.949],
#     'ADR': [869.959, 718.901, 590.566, 527.757, 379.855],
#     'RSLoRa': [418.612, 353.071, 309.995, 285.417, 250.312],
#     'DALoRa-Balance': [573.615, 551.314, 490.888, 461.912, 304.005],
#     'DALoRa-PDR': [617.031, 552.995, 535.476, 432.262, 335.355],
#     'DALoRa-EE': [412.128, 357.050, 381.230, 347.664, 291.953],
#     'DALoRa-Throughput': [887.723, 652.130, 428.261, 220.658, 201.981]

# }

radius_node_50_pdr_data = {
    'Random': [70.76, 59.89, 52.72, 48.59],
    'Round-Robin': [75.11, 62.51, 55.19, 51.21],
    'ADR': [80.41, 75.64, 70.22, 67.31],
    'RSLoRa': [66.62, 56.19,  49.33, 45.42],
    'DALoRa-Balance': [90.91, 89.83, 88.30, 85.81],
    'DALoRa-PDR': [95.30, 92.14, 89.46, 86.70],
    'DALoRa-EE': [84.14, 80.89, 78.73, 79.07],
    'DALoRa-TH': [89.91, 83.68,  83.68, 77.65]
    
}

radius_node_50_NetEE_data = {
    'Random': [46.32, 39.208, 34.512, 31.808],
    'Round-Robin': [53.064, 44.16, 38.992, 36.176],
    'ADR': [47.816, 35.112, 27.0, 23.624],
    'RSLoRa': [110.464, 68.992, 49.912, 42.208],
    'DALoRa-Balance': [84.216, 39.6, 22.328, 21.048],
    'DALoRa-PDR': [25.673, 23.469, 21.318, 17.253],
    'DALoRa-EE': [125.187, 50.790, 37.688, 23.146],
    'DALoRa-TH': [36.688, 26.077, 17.466, 8.863]
    
}

radius_node_50_Throughput_data = {
    'Random': [431.609, 365.315, 321.570, 296.404],
    'Round-Robin': [498.439, 414.796, 366.239, 339.808],
    'ADR': [869.959, 718.901, 590.566, 527.757],
    'RSLoRa': [418.612, 353.071, 309.995, 285.417],
    'DALoRa-Balance': [573.615, 551.314, 490.888, 461.912],
    'DALoRa-PDR': [617.031, 552.995, 535.476, 432.262],
    'DALoRa-EE': [412.128, 357.050, 381.230, 347.664],
    'DALoRa-TH': [887.723, 652.130, 428.261, 220.658]

}


radius_node_300_pdr_data = {
    'Random': [21.17, 19.61, 17.95, 17.00, 15.69],
    'Round-Robin': [23.61, 21.69, 19.92, 18.33, 16.83],
    'ADR': [38.19, 33.34, 28.82, 25.59, 22.23],
    'RSLoRa': [21.12, 19.88, 18.48, 17.54, 16.33],
    'DALoRa': [90.91, 89.83, 88.30, 85.81, 27.96]
}

radius_node_300_NetEE_data = {
    'Random': [13.620, 12.620, 11.550, 10.940, 10.097],
    'Round-Robin': [16.602, 15.995, 14.005, 12.892, 11.834],
    'ADR': [23.430, 15.995, 11.371, 8.914, 6.506],
    'RSLoRa': [28.290, 20.468, 15.251, 12.912, 10.735],
    'DALoRa': [10.527, 4.950, 2.791, 2.631, 9.819]
}

radius_node_300_Throughput_data = {
    'Random': [127.575, 118.208, 108.185, 102.474, 94.578],
    'Round-Robin': [154.819, 142.220, 130.604, 120.219, 110.360],
    'ADR': [417.942, 322.112, 246.166, 198.817, 149.754],
    'RSLoRa': [133.807, 125.914, 117.065, 111.123, 103.440],
    'DALoRa': [573.615, 551.314, 490.888, 461.912, 155.269]
}

# radius_NetPer_data = {
#     'Random': [1.698, 1.788, 1.84, 1.893, 2.041],
#     'Round-Robin': [1.884, 2.17, 2.038, 2.110, 2.223],
#     'ADR': [2.318, 2.244, 2.176, 2.360, 2.283],
#     'RSLoRa': [2.218, 2.124, 2.09, 2.082, 1.978],
#     'DALoRa': [2.422, 2.341, 2.279, 2.374, 2.220]
# }

# for key in radius_NetPer_data:
#     radius_NetPer_data[key] = [float(x / 3) for x in radius_NetPer_data[key]]


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

# num_NetPer_data = {
#     'Random': [1.750, 1.659, 1.50, 1.49, 1.548],
#     'Round-Robin': [1.922, 2.236, 1.78, 1.664, 1.777],
#     'ADR': [2.352, 2.395, 2.470, 2.576, 2.671],
#     'RSLoRa': [2.124, 2.000, 1.906, 1.884, 1.900],
#     'DALoRa': [2.341, 2.335, 2.579, 2.526, 2.590]
# }

# for key in num_NetPer_data:
#     num_NetPer_data[key] = [float(x / 3) for x in num_NetPer_data[key]]

# num_minimum_pdr_data = {
#     'Random': [59.92, 48.29, 33.47],
#     'Round-Robin': [46.64, 26.67, 5.23],
#     'ADR': [49.82, 48.30, 43.54],
#     'MAB': [78.32, 62.91, 45.03]
# }

# Network tropology radius
radius = [1000, 1500, 2000, 2500]

# Number of nodes
num_of_nodes = [50, 100, 150, 200, 250]

# 颜色和标签
colors_rgb = [(128,128,128), (243,177,105), (88,159,243), (167,210,186), (237,116,106),(237,116,106), (237,116,106), (237,116,106)]
colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]

makers = ['o', '+', 's', '^', '*', 'x', 'v', '<', '>', 'p']
# Define a list of hatch patterns to use
hatch_patterns = ['//', 'o', '\\\\']

labels = list(radius_node_300_pdr_data.keys())

'''Network PDR'''
plt.figure(figsize=(9, 9))  # 设置图形大小

bar_width = 40  # 柱子宽度
spacing = bar_width / 5  # 柱子间隔
num_algorithms = len(radius_node_300_pdr_data)

for i, (algo, pdr_values) in enumerate(radius_node_50_pdr_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量

    if i > 4:
        hatch_idx = (i-4) % len(hatch_patterns)
        hatch = hatch_patterns[hatch_idx]
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo, edgecolor='white', hatch=hatch, linewidth=0.2)
    else:
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], linewidth=0.4, label=algo)

plt.xlabel('Topology Radius (m)', fontsize=14)
plt.ylabel('Network PDR (%)', fontsize=14)
# plt.title('Network PDR for Different LoRa Parameter Allocation Algorithms')
plt.xticks([r + (num_algorithms * bar_width + (num_algorithms - 1) * spacing) / 2 for r in radius], radius, fontsize=12)
plt.yticks(range(40, 101, 10), fontsize=12)
plt.ylim(40, 100)
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
# fontP = FontProperties()
# fontP.set_size('xx-small')
plt.legend(fontsize='medium',loc='upper right')
 
# legend = plt.legend(fontsize='small')
# for text in legend.get_texts():
#     text.set_alpha(0.5)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)


plt.show()

figure_folder_path = "/home/uestc/LoRaSimulator/Joint-Parameter-Allocation-of-LoRaWAN-baesd-on-MARL/Figures"
if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
fig_name = 'TropologyRadius_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=600, bbox_inches='tight')


'''Network EE'''
plt.figure(figsize=(9, 9))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(radius_node_50_NetEE_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    if i > 4:
        hatch_idx = (i-4) % len(hatch_patterns)
        hatch = hatch_patterns[hatch_idx]
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo, edgecolor='white', hatch=hatch, linewidth=0.2)
    else:
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], linewidth=0.2, label=algo)

plt.xlabel('Topology Radius (m)', fontsize=14)
plt.ylabel('Network Energy Efficiency (bits/mJ)', fontsize=14)
# plt.title('Network Energy Efficiency for Different LoRa Parameter Allocation Algorithms')
plt.xticks([r + (num_algorithms * bar_width + (num_algorithms - 1) * spacing) / 2 for r in radius], radius, fontsize=12)
plt.yticks(range(0, 131, 10), fontsize=12)
plt.ylim(0, 130) 
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.05), ncol=8) 
plt.legend(fontsize='medium',loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkEE_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500, bbox_inches='tight')

'''Network Throughput'''
plt.figure(figsize=(9, 9))  # 设置图形大小

for i, (algo, pdr_values) in enumerate(radius_node_50_Throughput_data.items()):
    offset = bar_width * (num_algorithms - 1) / 2  # 计算每个算法的偏移量
    if i > 4:
        hatch_idx = (i-4) % len(hatch_patterns)
        hatch = hatch_patterns[hatch_idx]
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], label=algo, edgecolor='white', hatch=hatch, linewidth=0.2)
    else:
        plt.bar([r + i * (bar_width + spacing) - offset + bar_width / 2 for r in radius], pdr_values, width=bar_width, color=colors[i], linewidth=0.2, label=algo)

plt.xlabel('Topology Radius (m)', fontsize=14)
plt.ylabel('Network Throughput (bps)', fontsize=14)
# plt.title('Network Throughput for Different LoRa Parameter Allocation Algorithms')
plt.xticks([r + (num_algorithms * bar_width + (num_algorithms - 1) * spacing) / 2 for r in radius], radius, fontsize=12)
plt.yticks(range(0, 901, 100), fontsize=12)
plt.ylim(0, 900) 
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.legend(fontsize='medium',loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkThropughput_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500, bbox_inches='tight')


'''Network Performance'''
radius_NetPer_data = Net_Per_Index_Calculation(radius_node_50_pdr_data, radius_node_50_NetEE_data, radius_node_50_Throughput_data)
# print(radius_NetPer_data)


labels = list(radius_NetPer_data.keys())
data = list(radius_NetPer_data.values())

plt.figure(figsize=(12, 9))  # 设置图形大小

for i, (label, values) in enumerate(radius_NetPer_data.items()):
    plt.plot(radius, values, linestyle='-', marker=makers[i], markersize=5, color=colors[i], label=label,  linewidth=0.8)

plt.xlabel('Topology Radius (m)', fontsize=14)
plt.ylabel('Network Performance Index', fontsize=14)
# plt.title('Network Performance Index for Different LoRa Parameter Allocation Algorithms')
plt.xticks(radius)
plt.yticks([round(num, 3) for num in list(np.arange(0, 1.1, 0.10))])
plt.ylim(0, 1) 
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.legend(fontsize='medium', loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'TropologyRadius_NetworkPerformance_line_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500, bbox_inches='tight')


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
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkPDR_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=1000, bbox_inches='tight')

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
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkEE_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=1000, bbox_inches='tight')


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
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)    
plt.show()

fig_name = 'NumberofNodes_NetworkThroughput_bar_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500, bbox_inches='tight')


'''Network Performance'''
num_NetPer_data = Net_Per_Index_Calculation(num_pdr_data, num_NetEE_data, num_Throughput_data)
# print(num_NetPer_data)

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
plt.tick_params(axis='x', direction='in')  # x轴刻度线朝向图内
plt.tick_params(axis='y', direction='in')  # y轴刻度线朝向图内
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

fig_name = 'NumberofNodes_NetworkPerformance_line_chart.png'
plt.savefig(os.path.join(figure_folder_path, fig_name), dpi=500, bbox_inches='tight')

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


      
