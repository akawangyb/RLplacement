# --------------------------------------------------
# 文件名: multi_agent_reward
# 创建时间: 2024/2/29 14:51
# 描述: 分别显示placing_agent和routing_agent的回报
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt
# file_name = '../log_reserved/placing_routing_maddpg20240229-11_25_09OK/info.log'
file_name = '../log_reserved/placing_routing_per_maddpg20240302-17_07_26OK/info.log'


# 初始化存储数据的列表
episodes = []
total_rewards = []
agent_rewards = []  # 这是一个二维列表，用来存储每个智能体的奖励

# 读取并存储数据
with open(file_name, 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        episode_line = lines[i]
        reward_line = lines[i + 1]

        # 抽取episode和total reward
        parts = episode_line.split(',')
        episode = int(parts[0].split(':')[1].strip())
        total_reward = float(parts[1].split(':')[1].strip())

        # 抽取每个智能体的奖励
        rewards = eval(reward_line.split(':')[1].strip())

        # 将抽取的数据添加到相应的列表中
        episodes.append(episode)
        total_rewards.append(total_reward)
        agent_rewards.append(rewards)

# 创建第一张图：总奖励
plt.figure()
plt.plot(episodes, total_rewards)
plt.title('Total Rewards over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.show()

# 创建第二张图: 每个智能体的奖励
# num_agents = len(agent_rewards[0])
num_agents = 3
plt.figure()
for i in range(num_agents):
    rewards = [x[i] for x in agent_rewards]
    plt.plot(episodes, rewards, label=f'Agent {i+1}')
plt.title('Rewards of Each Agent over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Agent Rewards')
plt.legend()
plt.show()


# # 打开并读取文件
# with open(file_name, 'r') as f:
#     lines = f.readlines()
#
# # 初始化存储数据的列表
# episodes = []
# total_rewards = []
# agent_rewards = []  # 这是一个二维列表，用来存储每个智能体的奖励
#
# # 读取并存储数据
# for line in lines:
#     data = line.strip().split(',')
#     episodes.append(int(data[0]))
#     total_rewards.append(float(data[1]))
#     agent_rewards.append([float(x) for x in data[2:]])
#
# # 创建第一张图：total rewards
# plt.figure()
# plt.plot(episodes, total_rewards)
# plt.title('Total Rewards over Episodes')
# plt.xlabel('Episodes')
# plt.ylabel('Total Rewards')
# plt.show()
#
# # 这里假设你有13个智能体
# # 创建第二张图: 前三个智能体的奖励
# plt.figure()
# for i in range(3):
#     plt.plot(episodes, [x[i] for x in agent_rewards], label=f'Agent {i + 1}')
# plt.title('Rewards of First Three Agents over Episodes')
# plt.xlabel('Episodes')
# plt.ylabel('Agent Rewards')
# plt.legend()
# plt.show()
#
# 创建第三张图：剩下的智能体的奖励
plt.figure()
for i in range(3, 13):
    plt.plot(episodes, [x[i] for x in agent_rewards], label=f'Agent {i + 1}')
plt.title('Rewards of Other Agents over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Agent Rewards')
plt.legend()
plt.show()

