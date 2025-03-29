# --------------------------------------------------
# 文件名: ddpg_agent
# 创建时间: 2024/6/5 16:05
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import time
from collections import namedtuple

import numpy as np
import torch
import yaml

from ddpg import DDPG
from ddpg_memory import ReplayBuffer
from env_joint_place_distribution import EdgeDeploymentEnv
# from env import CustomEnv
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

env = EdgeDeploymentEnv('data')

# 输入状态包括,
# 服务器状态
# 边缘服务器数量 server_number     1
# 服务资源容量 4* server_number (n*4)
# 上一时刻的容器部署结果 (n*c)
#
# 当前的服务部署请求
# 每个请求包括其4种资源需求+容器类型
# 请求数量补全至单个时隙的最大请求数量 (r_max+5)
# *************************
# 当前时隙的编号\历史干扰因子均值
# *************************

# 状态空间
# N*4+C*N + R_max*7(4+类型+保活内存+延迟) + C*N(干扰情况)+时隙,服务器有5种资源

state_dim = env.state_dim

# R_max*(N+1)
action_dim = (env.max_requests, env.servers_number + 1)
replay_buffer = ReplayBuffer(state_size=state_dim,
                             action_size=action_dim,
                             buffer_size=config.buffer_size,
                             device=device)
agent = DDPG(state_dim=state_dim, action_dim=action_dim,
             hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
             tau=config.tau, gamma=config.gamma, device=device, log_dir=log_dir)


def evaluate(para_env, agent, i_episode, last_time, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = 0
    for _ in range(n_episode):
        state, done = env.reset()
        while not done:
            state = torch.Tensor(state).view(1, -1)
            x = agent.take_action(state, explore=False)
            y = get_placement(x)
            action = (x, y)
            state, rewards, done, info = env.step(action)  # rewards输出的是所有agent的和，info输出的是每个agent的reward
            rewards = torch.Tensor([rewards])
            returns += rewards.item()
    returns = returns * 1.0 / n_episode
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


def get_placement(action):
    routing_action = torch.argmax(action, dim=-1).tolist()
    x = [[0] * (env.servers_number + 1)] * env.max_requests
    for id, r in enumerate(routing_action):
        x[id][r] = 1
    # 根据x来求y
    y = [[0] * env.containers_number] * env.servers_number

    requests = env.request_generator.generate(env.current_timestep)
    # 遍历所有的服务器
    for n in range(env.servers_number):
        sum_d = 0
        required_containers = set()
        # for routing_id, r in enumerate(routing_action):
        for routing_id, r in enumerate(requests):
            if r == n: # 应该以在线的方式部署容器
                container_id = env.service_type_to_int[r['service_type']]
                required_containers.add(container_id)
                sum_d += r['mem_usage']
        sum_h = 0
        for container_id in required_containers:
            container_name = env.containers[id]
            for r in requests:
                if r['service_type'] == container_name:
                    sum_h += r['h_c']
                    break
        total_mem = sum_d + sum_h
        # 校验内存约束
        if total_mem > env.servers[n]['mem_capacity']:
            raise ValueError(f"服务器 {n} 内存不足（需{total_mem}）")
        # 设置必须加载的容器
        for c in required_containers:
            y[n][c] = 1
    return y


#

total_step = 0
best_reward = np.inf
last_time = time.time()
for i_episode in range(config.num_episodes):
    state, done = env.reset()  # 这里的state就是一个tensor
    episode_reward = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        x = agent.take_action(state, explore=True, eps=config.eps)
        # 解出的动作只有请求分配的的解,R_max*(N+1)
        # 下面根据这个解先求出哪些是必需要的加载的镜像
        y = get_placement(x)
        action = (x, y)
        next_state, rewards, done, info = env.step(action)
        rewards = torch.Tensor([rewards])
        transition = (state, x, rewards, torch.Tensor(next_state), done)
        replay_buffer.add(transition)
        state = next_state
        total_step += 1
        if replay_buffer.real_size >= config.minimal_size and total_step % config.update_interval == 0:
            batch = replay_buffer.sample(config.batch_size)
            agent.update(batch)
        episode_reward += rewards.item()
    record_info = 'Episode {} return: {}'.format(i_episode, episode_reward)
    print(record_info)
    f_log.write(record_info + '\n')
    if i_episode % 1 == 0:
        last_time = time.time()
        res = evaluate(para_env=env, agent=agent, i_episode=i_episode, last_time=last_time, n_episode=1)
        if res < best_reward:
            best_reward = res
            agent.save(log_dir + r'/ddpg_actor.pth')
