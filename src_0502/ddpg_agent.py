# --------------------------------------------------
# 文件名: ddpg_agent
# 创建时间: 2024/5/15 16:01
# 描述: ddpg_agent
# 作者: WangYuanbo
# --------------------------------------------------
# --------------------------------------------------
# 文件名: pmr_maddpg
# 创建时间: 2024/4/22 15:41
# 描述: maddpg算法+pmr
# 作者: WangYuanbo
# --------------------------------------------------
import time
from collections import namedtuple

import numpy as np
import torch
import yaml

from ddpg import DDPG
from ddpg_env import CustomEnv
from ddpg_memory import ReplayBuffer
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
                     'tau'
                     ])
config = Config(**config_data)

routing_action_dim = (env.user_number, env.container_number)
placing_action_dim = (env.server_number, env.container_number)
state_dim = env.state_dim

replay_buffer = ReplayBuffer(state_size=state_dim,
                             actions=[placing_action_dim, routing_action_dim],
                             buffer_size=config.buffer_size)

state_dims = [state_dim] * 2
action_dims = [placing_action_dim, routing_action_dim]
critic_input_dim = [state_dim, placing_action_dim, routing_action_dim]

agent = DDPG(state_dim=state_dim, action_dim=action_dims,
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
            raw_actions, env_action = agent.take_action(state, explore=False)
            state, rewards, done, info = env.step(env_action)  # rewards输出的是所有agent的和，info输出的是每个agent的reward
            # print(rewards)
            returns += rewards
    returns = returns / n_episode
    returns = returns.tolist()
    # 输出每个agent的reward，以及全体agent的和
    f_log.write("Episode: {}, total reward: {}\n".format(i_episode + 1, sum(returns)))
    f_log.write("each agent reward: {}\n".format(returns))
    print(f"Episode {i_episode + 1} : {returns}")
    print(f"sum of returns: {sum(returns)}")
    now_time = time.time()
    elapsed_time = round(now_time - last_time, 2)
    last_time = now_time
    print(elapsed_time)
    return returns


total_step = 0
best_reward = {
    'value': -np.inf,
    'list': [],
    'episode': -1,
}
last_time = time.time()
for i_episode in range(config.num_episodes):
    state, done = env.reset()  # 这里的state就是一个tensor
    while not done:
        state = state.view(1, -1)
        raw_actions, env_action = agent.take_action(state, explore=True)
        next_state, rewards, done, info = env.step(env_action)
        transition = (state, raw_actions, rewards, next_state, done)
        replay_buffer.add(transition)
        state = next_state
        total_step += 1
        # print(rewards)
        if replay_buffer.real_size >= config.minimal_size and total_step % config.update_interval == 0:
            batch = replay_buffer.sample(config.batch_size)
            # 要对记忆进行整形
            agent.update(batch)
    if (i_episode + 1) % 10 == 0:
        # ep_returns是一个np
        ep_returns = evaluate(env, agent, i_episode, last_time, n_episode=10)
        last_time = time.time()

        # # 保存最优的模型
        # if sum(ep_returns) > best_reward['value']:
        #     best_reward['value'] = sum(ep_returns)
        #     best_reward['list'] = ep_returns
        #     best_reward['episode'] = i_episode + 1
        #     maddpg.save()

print(best_reward['episode'], best_reward['value'], best_reward['list'])
