# --------------------------------------------------
# 文件名: tools
# 创建时间: 2024/7/17 14:56
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import os
import pickle
import re


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


# 定义一个函数读取文件并返回数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
        rewards_data = re.findall(r'Episode: (\d+), total reward: (\d+\.\d+)', log_data)
    episodes = [int(data[0]) for data in rewards_data]
    total_rewards = [-float(data[1]) / 1e6 for data in rewards_data]
    return episodes, total_rewards


def find_log_folders(folder_path):
    # folder_path = os.path.join(folder_path, 'log_res')
    ddpg = ''
    td3 = ''
    bc_td3 = ''
    bc_ddpg = ''
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹下的所有子文件夹
        for folder in dirs:
            if folder.startswith('ddpg'):
                ddpg = os.path.join(root, folder)
            elif folder.startswith('imitation_learning_env'):
                bc_ddpg = os.path.join(root, folder)
            elif folder.startswith('td3'):
                td3 = os.path.join(root, folder)
            # elif folder.startswith('imitation_learning_td3'):
            #     bc_td3 = os.path.join(root, folder)

    return ddpg, bc_ddpg, td3


# exp0 = 'exp0'
# exp1 = 'exp2'
# exp2 = 'b_exp'
#
# groups = [exp0, exp1, exp2]


def get_solutions_info(groups):
    groups_data = []
    for group in groups:
        file_path = r'../performance_res/' + group + '_compare_res.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        groups_data.append(data)
    return groups_data


def get_reward_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC']:
            continue
        labels.append(key)
        values.append(value[0] / 1e6)
        print(key, value[0])
    return labels, values


def get_dict_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC', 'Cloud']:
            continue
        labels.append(key)
        values.append(value[2])
        print(key, value[0])
    return labels, values


def get_delay_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC', 'Cloud']:
            continue
        labels.append(key)
        values.append(value[1])
        print(key, value[1])
    return labels, values

def get_time_data(exp_name='exp0'):
    # 假设有一个字典数据
    data = get_solutions_info([exp_name])
    data = data[0]
    # 提取字典键和值
    labels = []
    values = []
    for key, value in data.items():
        if key in ['DDPG', 'TD3', 'BC', 'Cloud']:
            continue
        labels.append(key)
        values.append(sum(value[-1])/len(value[-1]))
        print(len(value[-1]),sum(value[-1])/len(value[-1]))
        # print(key, value[-1])
    return labels, values



def read_critic(file_path):
    epochs = []
    loss = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("epoch"):
                epoch = int(line.split()[1])
                critic_loss = float(line.split()[4])/1e11
                epochs.append(epoch)
                loss.append(critic_loss)
    return epochs, loss


def find_name_log_folders(folder_path, agent_name):
    ele = []
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹下的所有子文件夹
        for folder in dirs:
            if folder.startswith(agent_name):
                ele.append(os.path.join(root, folder))

    return ele

