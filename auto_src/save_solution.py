# --------------------------------------------------
# 文件名: save_solution
# 创建时间: 2024/7/11 16:31
# 描述: 持久化模型的专家经验
# 作者: WangYuanbo
# --------------------------------------------------
import copy
import os
import pickle
from collections import namedtuple

import torch
import torch.nn.functional as F
import yaml

from baseline import random_rand
from env_with_interference import CustomEnv

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

Config = namedtuple('Config',
                    ['num_episodes',
                     'target_update',
                     'buffer_size',
                     'minimal_size',
                     'batch_size',
                     'actor_lr',
                     'critic_lr',
                     'update_interval',
                     'hidden_dim',
                     'gamma',
                     'tau',
                     'lmbda',
                     'epochs',
                     'eps',
                     'data_dir'
                     ])
config = Config(**config_data)

env = CustomEnv('cpu', config.data_dir)


def greedy(env: CustomEnv):
    """
    按个服务器试，能放就放，不能放换一个
    request 信息第一维是时间戳，第二维是容器id，第三维表示资源和延迟
    :param env:
    :return:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    container_info = env.container_info
    max_cap = [env.max_cpu, env.max_mem, env.max_net_in, env.max_net_out]
    action_list = []
    while not done:
        ts = env.timestamp
        routing_id = [env.server_number] * env.container_number
        server_cap = copy.deepcopy(env.server_info)
        for i in range(env.user_number):
            request = env.user_request_info[ts][i]
            for server_id in range(env.server_number):
                tag = True
                for j in range(4):
                    if server_cap[server_id][j] < request[j] or server_cap[server_id][j] <= 0 * (max_cap[j]):
                        tag = False
                        break
                # 检测存储
                if server_cap[server_id][4] < container_info[i][0]:
                    tag = False
                if tag:  # 证明当前服务器放得下
                    for j in range(4):
                        server_cap[server_id][j] -= request[j]
                        routing_id[i] = server_id
                    server_cap[server_id][4] -= container_info[i][0]
                    break
        action = torch.tensor(routing_id).long()
        action = F.one_hot(action, num_classes=env.server_number + 1)
        state, reward, done, info = env.step(action)
        action_list.append(action.tolist())

        total_reward += reward

    return sum(total_reward.tolist()), action_list


def baseline_gurobi_max_edge(env: CustomEnv, relax=False):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    action_list = []
    while not done:
        action = torch.tensor(env.model_solve_max_edge(relax=relax))
        # raw_action = torch.tensor(env.model_solve_max_edge(relax=relax))
        # while True:
        #     action = random_rand(raw_action)
        #     _, valid = env.cal_placing_rewards(action)
        #     if valid.all():
        #         break
        state, reward, done, info = env.step(action)
        action_list.append(action.tolist())
        total_reward += reward
    return sum(total_reward.tolist()), action_list


def baseline_gurobi(env: CustomEnv, relax=False):
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    cnt = 0
    action_list = []
    while not done:
        raw_action = torch.tensor(env.model_solve(relax))
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
        cnt += torch.sum(action[:, -1] == 1).item()
        state, reward, done, info = env.step(action)
        action_list.append(action.tolist())
        total_reward += reward
    print('cloud placing', cnt)
    return sum(total_reward.tolist()), action_list


max_edge = baseline_gurobi(env)
greedy_solve = greedy(env)
print(max_edge[0])
print(greedy_solve[0])
action_list = max_edge[-1] if max_edge[0] > greedy_solve[0] else greedy_solve[-1]

father_dir = 'data'
file_path = os.path.join(father_dir, config.data_dir + '_expert_solution.pkl')

with open(file_path, 'wb') as f:
    pickle.dump(action_list, f)

print('expert solution is saved in {}'.format(file_path))

# 从文件加载数组
# with open('model_para/expert_solution.pkl', 'rb') as f:
#     loaded_array = pickle.load(f)

# state, done = env.reset()
# total_reward = torch.zeros(env.container_number)
# while not done:
#     action = loaded_array[env.timestamp]
#     action = torch.tensor(action)
#     state, reward, done, info = env.step(action)
#     total_reward += reward
# print(total_reward)
# print(torch.sum(total_reward))
