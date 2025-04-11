# --------------------------------------------------
# 文件名: read_performance_result
# 创建时间: 2025/4/11 22:29
# 描述: 从文件中读取部署gurobi结果
# 作者: WangYuanbo
# --------------------------------------------------
# 从Pickle读取
import os
import pickle
from collections import namedtuple

import numpy as np
import yaml

from env_cache_sim_cal_delay import EdgeDeploymentEnv

with open('train_config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

Config = namedtuple('Config',
                    [
                        'GAMMA',
                        'LAMBDA',
                        'CLIP_EPS',
                        'LR_ACTOR',
                        'LR_CRITIC',
                        'BATCH_SIZE',
                        'EPOCHS',
                        'MAX_GRAD_NORM',
                        'C_MAX',
                        'data_dir',
                        'multiplier_lr',
                        'lag_multiplier',
                    ])
config = Config(**config_data)


def load_gurobi_pickle_results(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['deployment'], data['allocation']


# print(deployment)
# print(allocation)

# 把结果输入到env中
if __name__ == "__main__":
    gurobi_path = os.path.join(config.data_dir, 'deployment_results.pkl')
    deployment, allocation = load_gurobi_pickle_results(gurobi_path)
    env = EdgeDeploymentEnv(config.data_dir)
    # 模拟环境交互
    total_reward = 0
    total_ppo_reward = 0
    invalid_actions = 0
    invalid_load_actions = 0
    state, done = env.reset()
    while not done:
        print('ts', env.current_timestep)
        # 选择动作
        load_act = np.array(deployment[env.current_timestep])
        load_act = np.argmax(load_act, axis=-1)
        assign_act = np.array(allocation[env.current_timestep])
        # print(load_act.shape)
        # print(assign_act.shape)
        # print(len(env.current_requests))
        action = (load_act, assign_act)
        # 模拟环境返回奖励和新状态
        next_state, reward, done, info = env.step(action)
        reward = -reward
        # 存储经验
        total_reward += reward
    print('reward', total_reward)
