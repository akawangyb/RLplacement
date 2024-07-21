# --------------------------------------------------
# 文件名: experiment_epochs
# 创建时间: 2024/7/11 16:47
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import os.path
import pickle
import time
from collections import namedtuple

import numpy as np
import torch
import yaml

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


last_time = time.time()
best_reward = np.inf
# 训练前评估一次
evaluate(para_env=env, agent=agent, i_episode=0, last_time=last_time, n_episode=1)
num_epochs = 200

file_path = os.path.join('data', config.data_dir + '_expert_solution.pkl')
if not os.path.exists(file_path):
    raise 'expert solution does not exist'
# 从文件加载数组
with open(file_path, 'rb') as f:
    loaded_array = pickle.load(f)

state, done = env.reset()  # 这里的state就是一个tensor
transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
while not done:
    state = state.view(1, -1)
    action = torch.tensor(loaded_array[env.timestamp])
    next_state, rewards, done, info = env.step(action)
    next_state = next_state.view(1, -1)
    rewards = -rewards
    rewards = torch.sum(rewards)
    transition_dict['states'].append(state)
    transition_dict['actions'].append(action)
    transition_dict['rewards'].append(rewards)
    transition_dict['next_states'].append(next_state)
    transition_dict['dones'].append(done)

best_actor_path = os.path.join(log_dir, 'best_actor.pth')
for epoch in range(num_epochs):
    agent.learn(transition_dict, epoch=1)
    last_time = time.time()
    res = evaluate(para_env=env, agent=agent, i_episode=epoch + 1, last_time=last_time, n_episode=1)
    if res < best_reward:
        best_reward = res
        agent.save(best_actor_path)
