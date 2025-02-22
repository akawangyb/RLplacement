# --------------------------------------------------
# 文件名: save_solution
# 创建时间: 2024/7/11 16:31
# 描述: 持久化模型的专家经验
# 作者: WangYuanbo
# --------------------------------------------------
import copy
import os
import pickle
import time
from collections import namedtuple

import torch
import torch.nn.functional as F
import yaml

from baseline import random_rand
from ddpg import DDPG
from ddpg_memory import ReplayBuffer
from env_with_interference import CustomEnv
from tools import base_opt

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

f_log, log_dir, device = base_opt('env', config.data_dir)

env = CustomEnv(device=device, data_dir=config.data_dir)

state_dim = env.state_dim
action_dim = env.action_dim
replay_buffer = ReplayBuffer(state_size=state_dim,
                             action_size=action_dim,
                             buffer_size=config.buffer_size)

agent = DDPG(state_dim=state_dim, action_dim=action_dim,
             hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
             tau=config.tau, gamma=config.gamma, device=device, log_dir=log_dir)


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

    return action_list


def baseline_gurobi_max_edge(env: CustomEnv, relax=False):
    """
    用gurobi去解这个问题，优化目标是最大化边缘请求数量
    :param env:
    """
    state, done = env.reset()
    total_reward = torch.zeros(env.container_number)
    action_list = []
    while not done:
        raw_action = torch.tensor(env.model_solve_max_edge(relax=relax))
        while True:
            action = random_rand(raw_action)
            _, valid = env.cal_placing_rewards(action)
            if valid.all():
                break
        state, reward, done, info = env.step(action)
        action_list.append(action.tolist())
        total_reward += reward
    return action_list


def evaluate(para_env, agent, i_episode, last_time, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = torch.zeros(1)
    for _ in range(n_episode):
        state, done = env.reset()
        while not done:
            state = state.view(1, -1)
            action = agent.take_action(state, explore=False)
            state, rewards, done, info = env.step(action)  # rewards输出的是所有agent的和，info输出的是每个agent的reward
            rewards = torch.sum(rewards)
            returns += rewards
    returns = returns.item() / n_episode
    # 输出每个agent的reward，以及全体agent的和
    f_log.write("Episode: {}, total reward: {}\n".format(i_episode, returns))
    f_log.write("each agent reward: {}\n".format(returns))
    print(f"Episode {i_episode} : {returns}")
    print(f"sum of returns: {returns}")
    now_time = time.time()
    elapsed_time = round(now_time - last_time, 2)
    last_time = now_time
    print(elapsed_time)
    return returns

def merge_transition(transition1, transition2):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    transition_dict['states'] = transition1['states'] + transition2['states']
    transition_dict['actions'] = transition1['actions'] + transition2['actions']
    transition_dict['next_states'] = transition1['next_states'] + transition2['next_states']
    transition_dict['rewards'] = transition1['rewards'] + transition2['rewards']
    transition_dict['dones'] = transition1['dones'] + transition2['dones']
    return transition_dict


if __name__ == '__main__':
    max_edge = baseline_gurobi_max_edge(env)
    greedy_solve = greedy(env)

    state, done = env.reset()  # 这里的state就是一个tensor
    transition_dict1 = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    while not done:
        state = state.view(1, -1)
        action = torch.tensor(max_edge[env.timestamp])
        next_state, rewards, done, info = env.step(action)
        next_state = next_state.view(1, -1)
        rewards = -rewards
        rewards = torch.sum(rewards)
        transition_dict1['states'].append(state)
        transition_dict1['actions'].append(action)
        transition_dict1['rewards'].append(rewards)
        transition_dict1['next_states'].append(next_state)
        transition_dict1['dones'].append(done)

    transition_dict2 = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    state, done = env.reset()
    while not done:
        state = state.view(1, -1)
        action = torch.tensor(greedy_solve[env.timestamp])
        next_state, rewards, done, info = env.step(action)
        next_state = next_state.view(1, -1)
        rewards = -rewards
        rewards = torch.sum(rewards)
        transition_dict2['states'].append(state)
        transition_dict2['actions'].append(action)
        transition_dict2['rewards'].append(rewards)
        transition_dict2['next_states'].append(next_state)
        transition_dict2['dones'].append(done)

    father_dir = 'data'
    file_path1 = os.path.join(father_dir, config.data_dir + '_expert_solution.pkl1')
    file_path2 = os.path.join(father_dir, config.data_dir + '_expert_solution.pkl2')
    with open(file_path1, 'wb') as f:
        pickle.dump(max_edge, f)

    with open(file_path2, 'wb') as f:
        pickle.dump(greedy_solve, f)


    # 合并两组专家经验



    transition_dict = merge_transition(transition_dict1, transition_dict2)

    num_epochs = 200
    evaluate(para_env=env, agent=agent, i_episode=0, last_time=time.time(), n_episode=1)
    for epoch in range(num_epochs):
        # agent.learn(transition_dict1, critic_learn=0, epoch=1)
        # agent.learn(transition_dict2, critic_learn=0, epoch=1)
        agent.learn(transition_dict, epoch=1, critic_learn=0)
        last_time = time.time()
        res = evaluate(para_env=env, agent=agent, i_episode=epoch + 1, last_time=last_time, n_episode=1)
