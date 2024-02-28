# --------------------------------------------------
# 文件名: each_reward
# 创建时间: 2024/2/28 12:09
# 描述: 展示maddpg的每个智能体训练的奖励值
# 作者: WangYuanbo
# --------------------------------------------------
import ast

import matplotlib.pyplot as plt
import numpy as np

# 假设文件名是“rewards.txt”
filename = "../log_reserved/routing_agent20240228-12_19_41OK/info.log"

episodes = []
total_rewards = []
agent_rewards = []

with open(filename, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):  # 每次跳过一行
        # 解析 total reward
        line = lines[i].strip()  # 删除首尾空格
        parts = line.split(",")  # 使用逗号分割

        # 第一部分是"Episode"，第二部分是"total reward"
        episode = int(parts[0].split(" ")[1])
        total_reward = float(parts[1].split(": ")[1])

        episodes.append(episode)
        total_rewards.append(total_reward)

        # 解析 each agent reward
        line = lines[i + 1].strip()  # 删除首尾空格
        each_agent_reward = ast.literal_eval(line.split(": ")[1])  # 将字符串转换为列表

        agent_rewards.append(each_agent_reward)

# 转换为 np.array 方便处理
agent_rewards = np.array(agent_rewards)

# 创建折线图
plt.figure(figsize=(10, 6))

# 限制绘制的数据在前5000个episodes
episodes = episodes[:50]
total_rewards = total_rewards[:50]
agent_rewards = agent_rewards[:50, :]

# 绘制总的奖励曲线
plt.plot(episodes, total_rewards, label='Total Reward')

# 绘制每个 agent 的奖励曲线
for i in range(agent_rewards.shape[1]):
    plt.plot(episodes, agent_rewards[:, i], label=f'Agent {i + 1} Reward')

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards Over Episodes')
plt.legend(loc='upper right')
plt.grid()
plt.show()
