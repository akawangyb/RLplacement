# --------------------------------------------------
# 文件名: performance_test
# 创建时间: 2024/7/16 14:20
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import copy
import pickle
import time
from collections import namedtuple

import torch
import torch.nn.functional as F
import yaml

from auto_src.tools import find_solution_file
from baseline import random_rand
from ddpg import DDPG
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
dataset = '3exp_2'
env = CustomEnv('cpu', dataset)
state_dim = env.state_dim
action_dim = env.action_dim
agent = DDPG(state_dim=state_dim, action_dim=action_dim,
             hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
             tau=config.tau, gamma=config.gamma, device='cpu', log_dir='log')


def eval_model_solution(agent, is_valid=True):
    """
    测试用深度模型的性能
    :param agent:
    :return:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    episode_reward = []
    episode_interference = []
    episode_time = []
    while not done:
        state = state.view(1, -1)
        start_time = time.time()
        action = agent.take_action(state, explore=False)
        end_time = time.time()
        execution_time = end_time - start_time
        episode_time.append(execution_time)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        if not valid.all():
            print(env.timestamp, " not valid action")

        episode_reward.append(reward.tolist())
        episode_interference.append(factor.tolist())
        total_reward += reward

        state, reward, done, info = env.step(action)

    total_reward = torch.sum(total_reward).item()
    return total_reward, episode_reward, episode_interference, episode_time


def LR_Instant(env: CustomEnv, relax=True):
    """
    JSPRR
    :param env:
    :param relax:
    :return:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    episode_reward = []
    episode_interference = []
    episode_time = []
    while not done:
        start_time = time.time()
        raw_action = torch.tensor(env.model_solve(relax))
        time_0 = time.time()
        # print('time stamp', env.timestamp, time_0 - start_time)
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
            if time.time() - time_0 > 3:
                print(time.time() - time_0)
                action = torch.tensor(env.model_solve(False))
                break
        end_time = time.time()
        execution_time = end_time - start_time
        episode_time.append(execution_time)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        assert valid.all(), 'not valid action'

        episode_reward.append(reward.tolist())
        episode_interference.append(factor.tolist())
        total_reward += reward

        # 与环境交互
        state, reward, done, info = env.step(action)

    total_reward = torch.sum(total_reward).item()
    return total_reward, episode_reward, episode_interference, episode_time


def JSPRR(env: CustomEnv, relax=True):
    """
    测试JSPRR性能
    :param env:
    :param relax:
    :return:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    episode_reward = []
    episode_interference = []
    episode_time = []
    while not done:
        start_time = time.time()
        raw_action = torch.tensor(env.model_solve_max_edge(relax=relax))
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
        end_time = time.time()
        execution_time = end_time - start_time
        episode_time.append(execution_time)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        assert valid.all(), 'not valid action'

        episode_reward.append(reward.tolist())
        episode_interference.append(factor.tolist())
        total_reward += reward
        # 与环境交互
        state, reward, done, info = env.step(action)

    total_reward = torch.sum(total_reward).item()
    return total_reward, episode_reward, episode_interference, episode_time


def cloud(env: CustomEnv):
    user_request_info = env.user_request_info  # 是一个张量
    total_reward = 0
    episode_reward = []
    episode_interference = [[0] * env.container_number] * env.end_timestamp
    episode_time = [0] * env.end_timestamp

    for ts in range(env.end_timestamp):
        ts_lat = []
        for user_id in range(env.user_number):
            ts_lat.append(user_request_info[ts][user_id][-1].item() + env.cloud_delay)
        episode_reward.append(ts_lat)
        total_reward += sum(ts_lat)

    return total_reward, episode_reward, episode_interference, episode_time


def ideal(env: CustomEnv):
    user_request_info = env.user_request_info  # 是一个张量
    total_reward = 0
    episode_reward = []
    episode_interference = [[0] * env.container_number] * env.end_timestamp
    episode_time = [0] * env.end_timestamp

    for ts in range(env.end_timestamp):
        ts_lat = []
        for user_id in range(env.user_number):
            ts_lat.append(user_request_info[ts][user_id][-1].item() + env.edge_delay)
        episode_reward.append(ts_lat)
        total_reward += sum(ts_lat)

    return total_reward, episode_reward, episode_interference, episode_time


def greedy(env: CustomEnv, trade_factor=0.0):
    """
    按照best fit 原则，优先放置延迟大的。
    :param env:
    """
    total_reward = torch.zeros(env.container_number)
    episode_reward = []
    episode_interference = []
    episode_time = []

    state, done = env.reset()
    container_info = env.container_info

    max_cap = torch.tensor([env.max_cpu, env.max_mem, env.max_net_in, env.max_net_out])
    while not done:
        server_cap = copy.deepcopy(env.server_info)
        ts = env.timestamp
        routing_id = [env.server_number] * env.container_number
        start_time = time.time()
        for i in range(env.user_number):
            request = env.user_request_info[ts][i]
            for server_id in range(env.server_number):
                tag = True
                for j in range(4):
                    if server_cap[server_id][j] < request[j] or server_cap[server_id][j] <= trade_factor * max_cap[j]:
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
        end_time = time.time()
        execution_time = end_time - start_time
        episode_time.append(execution_time)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        if not valid.all():
            raise 'not valid action'

        episode_reward.append(reward.tolist())
        episode_interference.append(factor.tolist())
        total_reward += reward

        state, reward, done, info = env.step(action)
    total_reward = torch.sum(total_reward).item()
    return total_reward, episode_reward, episode_interference, episode_time


def greedy_place(env: CustomEnv, trade_factor=0.0):
    """
    按照 最大剩余优先 原则，优先放置延迟大的。
    :param env:
    """
    total_reward = torch.zeros(env.container_number)
    episode_reward = []
    episode_interference = []
    episode_time = []

    state, done = env.reset()
    container_info = env.container_info

    max_cap = [env.max_cpu, env.max_mem, env.max_net_in, env.max_net_out]
    while not done:
        server_cap = copy.deepcopy(env.server_info).tolist()
        for id, ele in enumerate(server_cap):
            ele.append(id)

        ts = env.timestamp
        routing_id = [env.server_number] * env.container_number
        start_time = time.time()

        for i in range(env.user_number):
            request = env.user_request_info[ts][i]
            server_cap = sorted(server_cap, key=lambda x: sum([x[i] / max_cap[i] for i in range(4)]), reverse=True)
            for id in range(env.server_number):
                server_id = server_cap[id][-1]
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

        action = torch.tensor(routing_id).long()
        action = F.one_hot(action, num_classes=env.server_number + 1)
        end_time = time.time()
        execution_time = end_time - start_time
        episode_time.append(execution_time)

        # 计算干扰因子
        reward, valid, factor = env.cal_placing_rewards(action, interference_factor=True)
        assert valid.all(), 'not valid action'

        episode_reward.append(reward.tolist())
        episode_interference.append(factor.tolist())
        total_reward += reward

        state, reward, done, info = env.step(action)
    total_reward = torch.sum(total_reward).item()
    return total_reward, episode_reward, episode_interference, episode_time


data = {

    'DDPG': [],
    'BC-DDPG': [],
    'TD3': [],
    'BC': [],
    'LR-Instant': [],
    'JSPRR': [],
    'Greedy': [],
    'Cloud': []
}

if __name__ == '__main__':
    father_dir = r'log_res/' + dataset

    _ = find_solution_file(father_dir)
    ddpg, bc_ddpg, td3, bc = _

    # td3
    agent.actor.load_state_dict(torch.load(td3))
    td3 = eval_model_solution(agent)
    print('td3 complete')
    # ddpg
    agent.actor.load_state_dict(torch.load(ddpg))
    ddpg = eval_model_solution(agent)
    print('ddpg complete')
    #
    # bc-ddpg
    agent.actor.load_state_dict(torch.load(bc_ddpg))
    bc_ddpg = eval_model_solution(agent)
    print('bc ddpg complete')

    # bc
    agent.actor.load_state_dict(torch.load(bc))
    bc = eval_model_solution(agent, False)
    print('bc complete')
    #
    greedy_res = greedy_place(env, )
    print(greedy_res[0])
    print('greedy complete')

    jsprr_res = JSPRR(env)
    print('jsprr complete')
    # #
    lr_ins = LR_Instant(env)
    print('lr instant complete')
    #
    cloud_res = cloud(env)
    print('cloud complete')
    #
    data['TD3'] = td3
    data['DDPG'] = ddpg
    data['BC-DDPG'] = bc_ddpg
    data['BC'] = bc
    data['JSPRR'] = jsprr_res
    data['Cloud'] = cloud_res
    data['LR-Instant'] = lr_ins
    data['Greedy'] = greedy_res
    for key, value in data.items():
        print(key, value[0])

    # 将数据写入JSON文件
    dir = r'performance_res/' + dataset + '_compare_res.pkl'
    with open(dir, "wb") as file:
        pickle.dump(data, file)

    ideal_res = ideal(env)
    print('ideal complete')
    print('ideal total delay', ideal_res[0])
