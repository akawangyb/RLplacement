# --------------------------------------------------
# 文件名: imitation_learning
# 创建时间: 2024/6/19 22:01
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import copy
import time
from collections import namedtuple

import torch
import yaml

from ddpg_memory import ReplayBuffer
from env_with_interference import CustomEnv
from td3 import TD3
# from env import CustomEnv
from tools import base_opt

env = CustomEnv('cpu')
f_log, log_dir, device = base_opt(env.name)
f_log.write('learning critic by using mse')
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

agent = TD3(state_dim=state_dim, action_dim=action_dim,
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
evaluate(para_env=env, agent=agent, i_episode=-1, last_time=last_time, n_episode=1)
actor_before = copy.deepcopy(agent.actor.state_dict())
for _ in range(1):
    state, done = env.reset()  # 这里的state就是一个tensor
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    action_set = []
    while not done:
        state = state.view(1, -1)
        action_set.append(env.model_solve(10))
        action = torch.tensor(action_set[0]).int()
        next_state, rewards, done, info = env.step(action)
        next_state = next_state.view(1, -1)
        rewards = -rewards
        rewards = torch.sum(rewards)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(rewards)
        transition_dict['next_states'].append(next_state)
        transition_dict['dones'].append(done)
    agent.learn(transition_dict, 40)
last_time = time.time()
evaluate(para_env=env, agent=agent, i_episode=1, last_time=last_time, n_episode=1)
actor_after = copy.deepcopy(agent.actor.state_dict())

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
        if replay_buffer.real_size >= config.minimal_size and total_step % config.update_interval == 0:
            batch = replay_buffer.sample(config.batch_size)
            agent.update(batch)
        episode_reward += rewards
    record_info = 'Episode {} return: {}'.format(i_episode, episode_reward.item())
    print(record_info)
    f_log.write(record_info + '\n')
    if i_episode % config.update_interval == 0:
        last_time = time.time()
        evaluate(para_env=env, agent=agent, i_episode=i_episode, last_time=last_time, n_episode=1)


def compare_weights(model_before, model_after, model_name):
    print("=========Comparing weights for {}=======".format(model_name))
    for (k1, v1), (k2, v2) in zip(model_before.items(), model_after.items()):
        assert k1 == k2
        print(f"For parameter {k1}, change is {(v2.float() - v1.float()).norm().item()}")

# compare_weights(actor_before, actor_after, 'actor')
