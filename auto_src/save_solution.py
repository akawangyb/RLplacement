# --------------------------------------------------
# 文件名: save_solution
# 创建时间: 2024/7/11 16:31
# 描述: 持久化模型的专家经验
# 作者: WangYuanbo
# --------------------------------------------------
import os
import pickle
import time
from collections import namedtuple

import torch
import yaml

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

env = CustomEnv('cpu', config.data_dir)
state, done = env.reset()
total_reward = torch.zeros(env.container_number)
action_list = []
while not done:
    print('time step: ', env.timestamp)
    last_time = time.time()
    action = env.model_solve()
    now_time = time.time()
    elapsed_time = round(now_time - last_time, 2)
    print(elapsed_time, 's')

    action_list.append(action)
    action = torch.tensor(action)
    state, reward, done, info = env.step(action)
    total_reward += reward

print(total_reward)
print(torch.sum(total_reward))

father_dir = 'data'
file_path = os.path.join(father_dir, config.data_dir + '_expert_solution.pkl')

with open(file_path, 'wb') as f:
    pickle.dump(action_list, f)

# 从文件加载数组
# with open('model_para/expert_solution.pkl', 'rb') as f:
#     loaded_array = pickle.load(f)

# state, done = env.reset()
# total_reward = torch.zeros(env.container_number)
# while not done:
#     action = loaded_array[env.timestamp]
#     action = torch.tensor(action)
#     state, reward, done, info = env.step(action)
#     total_reward += reward
# print(total_reward)
# print(torch.sum(total_reward))
