import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

def training_process(base_result_folder_path):
    subfolder_name = f"{nrNodes}_nodes_{radius}_m"
    MAConMAB_Config.result_folder_path = os.path.join(base_result_folder_path, subfolder_name)

    # 如果文件夹不存在则创建
    if not os.path.exists(MAConMAB_Config.result_folder_path):
        os.makedirs(MAConMAB_Config.result_folder_path)

    '''Network Energy Efficiency'''
    plt.figure()
    x = range(1,ILCMAB_Config.num_episode+1)
    plt.plot(x, MAConMAB_Config.NetworkEnergyEfficiency, label='NetworkEnergyEfficiency', color='blue')

    plt.title('Network Energy Efficiency changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency/(bits/mJ)')
    plt.legend()

    fig1_name = 'NetworkEnergyEfficiency_plot.png'
    plt.savefig(os.path.join(MAConMAB_Config.result_folder_path, fig1_name))

    '''Network PDR'''
    plt.figure()
    plt.plot(x, MAConMAB_Config.Network_PDR, '-', color='b')
    plt.title('Network PDR changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network PDR')

    fig2_name = 'Network_PDR_plot.png'
    plt.savefig(os.path.join(MAConMAB_Config.result_folder_path, fig2_name))

    '''Network Throughput'''
    plt.figure()
    plt.plot(x, MAConMAB_Config.Network_Throughput, '-', color='b')
    plt.title('Network throughput changes with increasing episode')
    plt.xlabel('Episode')
    plt.ylabel('Network Throughput')

    fig3_name = 'Network_Throughput.png'
    plt.savefig(os.path.join(MAConMAB_Config.result_folder_path, fig3_name))



def heatmap():
    '''Heatmap'''
    # prepare show
    plt.figure(figsize=(6, 6))
    ax = plt.gcf().gca()
    legend_elements = []
    flag = 0

    for node in nodes:
        graphics_node(node,ax)
        if node.bs.id == 0 and flag == 0:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Node', markerfacecolor='blue', markersize=6))
            flag = 1
    for GW in bs:
        graphics_gateway(GW,ax)
        if GW.id == 0:
            legend_elements.append(Line2D([0], [0], marker='^', color='w', label='Gateway', markerfacecolor='red', markersize=8))

    # 保证生成的图像是正方形
    ax.set_aspect('equal')

    plt.xlabel('Distance (m)')
    plt.ylabel('Distance (m)')
    plt.xlim(-(radius+100), radius+100)
    plt.ylim(-(radius+100), radius+100)
    
    plt.tick_params(axis='x', direction='in')  
    plt.tick_params(axis='y', direction='in')  
    
    # 创建虚拟 ScalarMappable 对象来生成颜色条
    sm = ScalarMappable(cmap=cm.plasma)
    sm.set_array([])  # 设置一个空数组，因为颜色映射实际上不需要数据

    # 添加颜色条到图像
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Throughput')  # 设置颜色条的标签

    # 添加图例
    ax.legend(handles=legend_elements, loc='upper right')

    if storage_flag == 1:  
        fig_name = 'network_tropology.png'
        plt.savefig(os.path.join(ParameterConfig.result_folder_path, fig_name), dpi=800, bbox_inches='tight')   