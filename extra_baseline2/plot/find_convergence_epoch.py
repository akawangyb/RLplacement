# --------------------------------------------------
# 文件名: find_convergence_epoch
# 创建时间: 2025/3/23 13:43
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
# baseline
import re

import numpy as np

from tools import find_log_folders


def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [-float(data[1]) for data in rewards_data]
    return episodes, total_rewards


def sliding_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size:
            smoothed_data.append(data[i])
        else:
            window_values = data[i - window_size:i]
            average = sum(window_values) / window_size
            smoothed_data.append(average)
    return smoothed_data


def find_convergence_episode(rewards, window_size=50, threshold=0.05):
    """
    检测初次收敛的回合数
    :param rewards: 累计奖励列表，格式为[episode_1_reward, episode_2_reward, ...]
    :param window_size: 判定收敛的连续回合数窗口，默认为50
    :param threshold: 波动阈值，默认为5%
    :return: 初次收敛的回合编号（从1开始计数），若未收敛返回-1
    """
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i + window_size]

        # 计算窗口内奖励的均值和标准差
        mean = np.mean(window)
        std = np.std(window)

        # 避免均值为0导致计算错误（当奖励全为0时认为已收敛）
        if np.isclose(mean, 0):
            return i + 1  # 返回起始回合编号

        # 计算变异系数（Coefficient of Variation）
        cv = std / abs(mean)

        if cv <= threshold:
            return i + 1  # 返回起始回合编号

    return -1  # 未找到收敛窗口


def find_global_first_convergence(rewards, window_size=50, threshold=0.05, look_ahead=10):
    """
    检测全体回合中首次达到并稳定在收敛值的回合
    :param rewards: 累计奖励列表
    :param window_size: 收敛窗口大小（用于计算收敛均值）
    :param threshold: 波动阈值（默认5%）
    :param look_ahead: 后续稳定性验证回合数（默认10回合）
    :return: (收敛均值, 首次稳定回合编号)
    """
    # Step 1: 确定收敛窗口及收敛均值
    convergence_start = -1
    convergence_mean = None
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i + window_size]
        cv = np.std(window) / abs(np.mean(window))
        if cv <= threshold:
            convergence_start = i
            convergence_mean = np.mean(window)
            break

    if convergence_start == -1:
        return (None, -1)  # 未检测到收敛

    # Step 2: 全局搜索首次达到收敛均值的稳定回合
    for ep in range(len(rewards) - look_ahead + 1):
        current_reward = rewards[ep]
        # 检查单点稳定性（与收敛均值的偏差≤5%）
        if abs(current_reward - convergence_mean) / abs(convergence_mean) > threshold:
            continue

        # 检查后续稳定性（后续look_ahead回合的波动≤5%）
        subsequent = rewards[ep:ep + look_ahead]
        if np.std(subsequent) / abs(np.mean(subsequent)) <= threshold:
            return (convergence_mean, ep + 1)  # 回合编号从1开始

    return (convergence_mean, -1)  # 未找到稳定点


# father_dir = r'../log_res/4exp_3'
# file_paths = find_name_log_folders(father_dir, agent_name='imitation_learning')
# print(file_paths)
# label = [ele[-6:] for ele in file_paths]
father_dir = r'../log_res/5exp_1'
file_paths = find_log_folders(father_dir)
label = ['TD3', 'DDPG', 'BC-DDPG', ]

# print(file_paths)

# 文件路径列表

file_paths = [ele + '/output_info.log' for ele in file_paths]

for i, file_path in enumerate(file_paths):
    # if i>=2:
    #     break
    episodes, total_rewards = read_data(file_path)
    # total_rewards = sliding_average(total_rewards, 5)
    # print(len(episodes), len(total_rewards))
    # print(total_rewards)
    mean_epoch = find_convergence_episode(total_rewards, 93, 0.03)
    mean_value, first_epoch = find_global_first_convergence(total_rewards, 93, 0.03)
    print(file_path)
    print(first_epoch, mean_epoch, mean_value)
