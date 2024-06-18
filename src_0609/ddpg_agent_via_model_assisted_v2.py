# --------------------------------------------------
# 文件名: ddpg_agent
# 创建时间: 2024/6/5 16:05
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import math
import random
import time
from collections import namedtuple

import numpy as np
import torch
import yaml

from ddpg import DDPG
from ddpg_memory import ReplayBuffer
from env_with_interference import CustomEnv
from tools import base_opt

env = CustomEnv('cpu')
f_log, log_dir, device = base_opt(env.name)
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
                     'eps'
                     ])
config = Config(**config_data)

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


best_reward = {
    'value': -np.inf,
    'list': [],
    'episode': -1,
}
last_time = time.time()


def linear_decay(step, total_epoch=20, init_value=1):
    if step >= total_epoch:
        return 0
    """
    设置在step轮，应该从模型中取得记忆的比例
    :param step:
    """
    n = total_epoch  # 设置总轮数
    decay_rate = init_value * 1.0 / n  # 设置每轮的衰减量

    value = init_value - decay_rate * step
    return value


def exp_decay(step, init_value=1, decay_rate=0.01):
    """
    设置在step轮，应该从模型中取得记忆的比例
    :param step:
    """
    value = init_value * math.exp(-decay_rate * step)
    return value



total_step = 0
for i_episode in range(config.num_episodes):
    # for i_episode in range(1):
    state, done = env.reset()  # 这里的state就是一个tensor
    episode_reward = torch.zeros(1)
    while not done:
        state = state.view(1, -1)
        p = random.random()
        p_std = linear_decay(i_episode, total_epoch=200, init_value=0.8)
        action = None
        if p <= p_std:
            # 采用gurobi模型产生的动作
            raw_action = env.model_solve_relax()
            Flag = False
            for _ in range(20):
                # print(env.timestamp)
                raw_action = env.random_rand(raw_action)
                action = torch.tensor(raw_action).int()
                _, valid = env.cal_placing_rewards(action)
                if valid.all():
                    Flag = True
                    break
            if Flag == False:
                action = torch.tensor(env.model_solve()).int()
        else:
            # 采用ddpg模型产生的动作
            action = agent.take_action(state, explore=True)
        next_state, rewards, done, info = env.step(action)
        rewards = torch.sum(rewards)
        rewards = -rewards
        transition = (state, action, rewards, next_state, done)
        replay_buffer.add(transition)
        state = next_state
        total_step += 1
        if total_step % config.update_interval == 0 and replay_buffer.real_size >= config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            agent.update(batch)
        episode_reward += rewards
    record_info = 'Episode {} return: {}'.format(i_episode, episode_reward.item())
    print(record_info)
    f_log.write(record_info + '\n')
    # if i_episode % config.update_interval == 0:
    #     last_time = time.time()
    #     evaluate(para_env=env, agent=agent, i_episode=i_episode, last_time=last_time)
