# --------------------------------------------------
# 文件名: plot
# 创建时间: 2024/6/19 12:14
# 描述: 从日志中读取数据，用图片显示出来
# 作者: WangYuanbo
# --------------------------------------------------


import re

import matplotlib.pyplot as plt


# 定义一个函数读取文件并返回数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [-float(data[1]) for data in rewards_data]
    return episodes, total_rewards


file_path1 = r'log_res/ddpg_agent_env-20240623-211208/output_info.log'
file_path2 = r'log_res/td3_agent_env-20240623-211214/output_info.log'
file_path3 = r'log_res/imitation_learning_td3_env-20240623-211220/output_info.log'
# 文件路径列表
file_paths = [file_path1, file_path2, file_path3]
label = ['ddpg_baseline', 'td3', 'BC_TD3']

# 创建一个新的figure
plt.figure()

# 对文件路径列表中的每个文件进行处理
for i, file_path in enumerate(file_paths):
    episodes, total_rewards = read_data(file_path)

    # 绘制total reward的折线图
    plt.plot(episodes[:100], total_rewards[:100], marker='.', label=label[i])

# 添加图表标题和坐标轴标签
plt.title('Total Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# 添加图例
plt.legend()

# 显示图形
plt.show()
