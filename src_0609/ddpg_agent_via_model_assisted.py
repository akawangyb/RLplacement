# --------------------------------------------------
# 文件名: ddpg_agent
# 创建时间: 2024/6/5 16:05
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import time
from collections import namedtuple
from copy import deepcopy

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

model_replay_buffer = deepcopy(replay_buffer)

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


def linear_decay(step, start_epoch=0, total_epoch=20, init_value=1):
    if step < start_epoch:
        return init_value
    if step >= total_epoch + start_epoch:
        return 0

    """
    设置在step轮，应该从模型中取得记忆的比例
    :param step:
    """
    n = total_epoch  # 设置总轮数
    decay_rate = init_value * 1.0 / n  # 设置每轮的衰减量

    value = init_value - decay_rate * (step - start_epoch)
    return value


for i_episode in range(12):
    state, done = env.reset()  # 这里的state就是一个tensor
    episode_reward = torch.zeros(1)
    while not done:
        action = env.model_solve()
        action = torch.tensor(action).int()
        # while True:
        #     action = random_rand(action)
        #     action = torch.tensor(action)
        #     _, valid = env.cal_placing_rewards(action)
        #     if valid.all():
        #         break
        next_state, rewards, done, info = env.step(action)
        rewards = torch.sum(rewards)
        rewards = -rewards
        episode_reward += rewards
        transition = (state, action, rewards, next_state, done)
        model_replay_buffer.add(transition)
        state = next_state
    record_info = 'Episode {} return: {}'.format(i_episode, episode_reward.item())
    print(record_info)
print(model_replay_buffer.real_size)

total_step = 0
for i_episode in range(config.num_episodes):
    # for i_episode in range(1):
    state, done = env.reset()  # 这里的state就是一个tensor
    episode_reward = torch.zeros(1)
    while not done:
        state = state.view(1, -1)
        action = agent.take_action(state, explore=True)
        next_state, rewards, done, info = env.step(action)
        rewards = torch.sum(rewards)
        rewards = -rewards
        transition = (state, action, rewards, next_state, done)
        replay_buffer.add(transition)
        state = next_state
        total_step += 1
        if total_step % config.update_interval == 0:
            # 假设前50轮内设置一个线性衰减的参数
            model_batch_size = int(
                config.batch_size * linear_decay(i_episode, start_epoch=100, total_epoch=10, init_value=1))
            agent_batch_size = config.batch_size - model_batch_size

            model_memory = model_replay_buffer.sample(model_batch_size)
            agent_memory = replay_buffer.sample(agent_batch_size)


            # 记忆整合到一起
            def merger_memory(mem1, mem2):
                s1, a1, r1, s1_, d1 = mem1
                s2, a2, r2, s2_, d2 = mem2
                s = torch.cat([s1, s2])
                a = torch.cat([a1, a2])
                r = torch.cat([r1, r2])
                s_ = torch.cat([s1_, s2_])
                d = torch.cat([d1, d2])
                return s, a, r, s_, d


            batch = merger_memory(model_memory, agent_memory)
            agent.update(batch)
        episode_reward += rewards
    record_info = 'Episode {} return: {}'.format(i_episode, episode_reward.item())
    print(record_info)
    f_log.write(record_info + '\n')
    # if i_episode % config.update_interval == 0:
    #     last_time = time.time()
    #     evaluate(para_env=env, agent=agent, i_episode=i_episode, last_time=last_time)
