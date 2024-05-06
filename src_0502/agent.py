# --------------------------------------------------
# 文件名: pmr_maddpg
# 创建时间: 2024/4/22 15:41
# 描述: maddpg算法+pmr
# 作者: WangYuanbo
# --------------------------------------------------
import argparse
import os
import random
import sys
import time
from collections import namedtuple
from datetime import datetime

import numpy as np
import torch
import yaml

from env import CustomEnv
from maddpg import MADDPG
from memory.buffer import MultiAgentReplayBuffer

# 指定训练gpu
parser = argparse.ArgumentParser(description='选择训练GPU的参数')
parser.add_argument('--gpu', type=int, default=0, help='要使用的GPU的编号')

# 解析参数
args = parser.parse_args()

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# 确认使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
env = CustomEnv(device=device)

# 获得脚本名字
filename = os.path.basename(sys.argv[0])  # 获取脚本文件名
script_name = os.path.splitext(filename)[0]  # 去除.py后缀
# 日志输出路径
father_log_directory = 'log'
if not os.path.exists(father_log_directory):
    os.makedirs(father_log_directory)
current_time = datetime.now()
formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
log_file_name = script_name + '_' + env.name + '-' + formatted_time

log_path = os.path.join(father_log_directory, log_file_name)
# 规范化文件路径
log_dir = os.path.normpath(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'output_info.log')
f_log = open(log_path, 'w', encoding='utf-8')

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

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

config = Config(**config_data)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

routing_action_dims = env.routing_action_dims
placing_action_dims = env.placing_action_dims
action_dims = placing_action_dims + routing_action_dims
state_dims = env.state_dims
critic_input_dim = env.state_dim + sum(action_dims)

replay_buffer = MultiAgentReplayBuffer(
    state_size=env.state_dim,
    actions_size=action_dims,
    device=device,
    buffer_size=config.buffer_size,
)

maddpg = MADDPG(env=env, action_dims=action_dims, state_dims=state_dims,
                critic_input_dim=critic_input_dim,
                hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
                tau=config.tau, gamma=config.gamma, device=device, log_dir=log_dir)


def evaluate(para_env, maddpg, n_episode=10):
    # 对学习的策略进行评估,此时不会进行探索
    env = para_env
    returns = torch.zeros(env.agents_number)
    for _ in range(n_episode):
        state, done = env.reset()
        while not done:
            actions = maddpg.take_action(state, explore=False)
            placing_actions = actions[:env.placing_agents_number]
            routing_actions = actions[-env.routing_agents_number:]
            placing_actions = torch.cat(placing_actions, dim=0)
            placing_actions = (placing_actions > 0.5).int()
            routing_actions = [torch.argmax(routing_action) for routing_action in routing_actions]
            routing_actions = torch.stack(routing_actions)

            env_action = {
                'placing_action': placing_actions,
                'routing_action': routing_actions,
            }
            state, rewards, done, info = env.step(env_action)  # rewards输出的是所有agent的和，info输出的是每个agent的reward
            returns += info * 1.0
    returns = returns / n_episode
    returns = returns.tolist()
    # 输出每个agent的reward，以及全体agent的和
    f_log.write("Episode: {}, total reward: {}\n".format(i_episode + 1, sum(returns)))
    f_log.write("each agent reward: {}\n".format(returns))
    print(f"Episode {i_episode + 1} : {returns}")
    print(f"sum of returns: {sum(returns)}")
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
        actions = maddpg.take_action(state, explore=True)
        placing_actions = actions[:env.placing_agents_number]
        # 把placing_action转换成二维张量
        placing_actions = torch.cat(placing_actions, dim=0).float()
        placing_actions = (placing_actions > 0.5).int()
        routing_actions = actions[-env.routing_agents_number:]
        routing_actions = [torch.argmax(routing_action) for routing_action in routing_actions]
        routing_actions = torch.stack(routing_actions)
        # routing_actions = torch.cat(routing_actions, dim=0)
        # print("placing_actions:", placing_actions)
        # print("routing_actions:", routing_actions)
        env_action = {
            'placing_action': placing_actions,
            'routing_action': routing_actions,
        }

        # 从actions里面获得env可以识别的action
        # 这里输出的actions是一个list的np array
        # 主要是这一步，action与环境进行交互
        next_state, rewards, done, info = env.step(env_action)
        # print("rewards:", rewards.long())
        actions = env_action

        replay_buffer.add((state, actions, rewards, next_state, done))
        state = next_state
        total_step += 1
        if replay_buffer.real_size >= config.minimal_size and total_step % config.update_interval == 0:
            batch = replay_buffer.sample(config.batch_size)
            for i in range(maddpg.placing_agents_number + maddpg.routing_agents_number):
                maddpg.update(i, batch)
            maddpg.update_all_targets()
    # break
    if (i_episode + 1) % 10 == 0:
        # ep_returns是一个np
        ep_returns = evaluate(env, maddpg, n_episode=10)
        now_time = time.time()
        elapsed_time = round(now_time - last_time, 2)
        last_time = now_time
        print(elapsed_time)
        # 保存最优的模型
        if sum(ep_returns) > best_reward['value']:
            best_reward['value'] = sum(ep_returns)
            best_reward['list'] = ep_returns
            best_reward['episode'] = i_episode + 1
            maddpg.save()

print(best_reward['episode'], best_reward['value'], best_reward['list'])
