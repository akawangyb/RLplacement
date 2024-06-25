# --------------------------------------------------
# 文件名: baseline
# 创建时间: 2024/6/6 15:04
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import random

import torch
import torch.nn.functional as F

from env_with_interference import CustomEnv


def baseline_cloud(env: CustomEnv):
    # 获得ts时刻的用户请求
    # 6分别是【ts】[user_id]imageID,cpu,mem,net-in,net-out,lat
    user_request_info = env.user_request_info  # 是一个张量
    lat = []
    for ts in range(env.end_timestamp):
        ts_lat = 0
        for user_id in range(env.user_number):
            ts_lat += user_request_info[ts][user_id][-1].item() + env.cloud_delay
        lat.append(ts_lat)

    return sum(lat)


def baseline_greedy_placement(env: CustomEnv):
    '''
    按照贪婪的模式去选择部署，
    :param env:
    :return:
    '''
    placed_position = [-1] * env.container_number  # 每个容器部署在什么地方
    server_storage_supply = env.server_info[:, 4]
    server_id = 0
    for container_id in range(env.container_number):
        storage_demand = env.container_info[container_id, 0]
        if server_storage_supply[server_id] >= storage_demand:
            server_storage_supply[server_id] -= storage_demand
            placed_position[container_id] = server_id
        else:
            server_id += 1
        if server_id >= env.server_number:
            break
    # 处理没放下的容器
    for container_id, server_id in enumerate(placed_position):
        if server_id == -1:
            placed_position[container_id] = env.server_number

    state, done = env.reset()
    action = torch.tensor(placed_position, dtype=torch.long)
    # action = torch.nn.functional.one_hot(action, env.server_number + 1)
    total_reward = torch.zeros(env.container_number)
    while not done:
        temp_action = action.clone().detach()
        onehot_action = F.one_hot(temp_action, env.server_number + 1)
        reward, is_valid_placing = env.cal_placing_rewards(placing_action=onehot_action)
        if not torch.all(is_valid_placing):
            temp_action[~is_valid_placing] = env.server_number
        onehot_action = F.one_hot(temp_action, env.server_number + 1)
        state, reward, done, info = env.step(onehot_action)
        total_reward += reward
    return total_reward


def random_rand(action):
    res = []
    for con in action:
        res_con = []
        for ele in con:
            # print(ele)
            p = 1 if random.random() >= ele else 0
            res_con.append(p)
        res.append(res_con)
    return res


def baseline_gurobi(env: CustomEnv):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    while not done:
        action = env.model_solve(10)
        action = torch.tensor(action[6])
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def baseline_gurobi_max_edge(env: CustomEnv):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    while not done:
        action = env.model_solve_max_edge()
        action = torch.tensor(action)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def baseline_gurobi_relax(env: CustomEnv):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    while not done:
        raw_action = env.model_solve_relax()
        Flag = False
        for _ in range(20):
            action = random_rand(raw_action)
            action = torch.tensor(action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                Flag = True
                break
        if Flag == False:
            action = torch.tensor(env.model_solve()[0]).int()
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    env = CustomEnv('cpu')
    res1 = baseline_gurobi(env=env)
    res1 = torch.sum(res1)
    res2 = baseline_cloud(env=env)
    res3 = baseline_gurobi_relax(env=env)
    res3 = torch.sum(res3)

    res4 = baseline_gurobi_max_edge(env=env)
    res4 = torch.sum(res4)

    print('gurobi integer', res1)
    print('gurobi relax', res3)
    print('gurobi integer max edge', res4)
    print('cloud', res2)
