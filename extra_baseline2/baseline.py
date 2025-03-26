import copy
import time

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
        # print(ts_lat)

    return sum(lat)


def random_rand(action):
    random_tensor = torch.rand_like(action)
    # 按照条件进行元素赋值
    result_tensor = torch.where(random_tensor > action, torch.tensor(0), torch.tensor(1))
    return result_tensor


def baseline_gurobi(env: CustomEnv, relax=False):
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    cnt = 0
    while not done:
        raw_action = torch.tensor(env.model_solve(relax))
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
        cnt += torch.sum(action[:, -1] == 1).item()
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def baseline_gurobi_max_edge(env: CustomEnv, relax=False):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    cnt = 0
    while not done:
        raw_action = torch.tensor(env.model_solve_max_edge(relax=relax))
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
        cnt += torch.sum(action[:, -1] == 1).item()
        _, valid = env.cal_placing_rewards(action)
        assert torch.all(valid), f'exist fuck action {valid}'
        state, reward, done, info = env.step(action)
        total_reward += reward
    print(cnt)
    return total_reward


def baseline_sum_delay(env: CustomEnv):
    n = env.container_number
    cnt = 0
    for i in range(env.end_timestamp):
        res = 0
        for j in range(n):
            res += env.user_request_info[i][j][-1]
        cnt += res
    return cnt, env.edge_delay * env.container_number * env.end_timestamp


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
    while not done:
        ts = env.timestamp
        routing_id = [env.server_number] * env.container_number
        server_cap = copy.deepcopy(env.server_info)
        for i in range(env.user_number):
            request = env.user_request_info[ts][i]
            for server_id in range(env.server_number):
                tag = True
                for j in range(4):
                    if server_cap[server_id][j] < request[j] or server_cap[server_id][j] <= 0.0 * (max_cap[j]):
                        # if server_cap[server_id][j] < request[j]:
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
        total_reward += reward
    return total_reward

def greedy_place(env: CustomEnv, trade_factor=0.2):
    """
    按照 最大剩余优先 原则，优先放置延迟大的。
    :param env:
    """
    total_reward = torch.zeros(env.container_number)

    state, done = env.reset()
    container_info = env.container_info

    max_cap = [env.max_cpu, env.max_mem, env.max_net_in, env.max_net_out]
    while not done:
        server_cap = copy.deepcopy(env.server_info).tolist()
        for id, ele in enumerate(server_cap):
            ele.append(id)

        ts = env.timestamp
        routing_id = [env.server_number] * env.container_number

        for i in range(env.user_number):
            request = env.user_request_info[ts][i]
            server_cap = sorted(server_cap, key=lambda x: sum([x[i] / max_cap[i] for i in range(4)]), reverse=True)
            for id in range(env.server_number):
                server_id = server_cap[id][-1]
                # print(server_id)
                tag = True
                for j in range(4):
                    if server_cap[id][j] < request[j] or server_cap[id][j] <= trade_factor * max_cap[j]:
                        tag = False
                        break
                # 检测存储
                if server_cap[id][4] < container_info[i][0]:
                    tag = False
                if tag:  # 证明当前服务器放得下
                    for j in range(4):
                        server_cap[id][j] -= request[j]
                        routing_id[i] = server_id
                    server_cap[id][4] -= container_info[i][0]
                    break
        # print(routing_id)

        action = torch.tensor(routing_id).long()
        action = F.one_hot(action, num_classes=env.server_number + 1)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        assert valid.all(), 'not valid action'
        total_reward += reward

        state, reward, done, info = env.step(action)
    return total_reward


if __name__ == '__main__':
    env = CustomEnv('cpu', data_dir='1exp_1')

    # res1 = baseline_gurobi(env=env)
    # res1 = torch.sum(res1)
    # print('JSPRR', res1)
    #
    res2 = baseline_gurobi(env=env, relax=True)
    res2 = torch.sum(res2)
    print('JSPRR', res2)

    # res4 = baseline_gurobi_max_edge(env=env)
    # res4 = torch.sum(res4)
    # print('gurobi max edge', res4)
    #
    res5 = baseline_gurobi_max_edge(env=env, relax=True)
    res5 = torch.sum(res5)
    print('LR-Instant', res5)

    # res6 = baseline_cloud(env=env)
    # print('cloud', res6)
    #
    # res7 = greedy_place(env,0)
    # res7 = torch.sum(res7)
    # print('greedy', res7)

    res8 = greedy(env)
    res8 = torch.sum(res8)
    print('greedy', res8)

    print('total computing delay', baseline_sum_delay(env=env))
