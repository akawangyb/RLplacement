# --------------------------------------------------
# 文件名: performance_test_v2
# 创建时间: 2024/12/30 10:51
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import pickle
from collections import namedtuple

import torch
import yaml

from ddpg import DDPG
from env_with_interference import CustomEnv
from performance_test import eval_model_solution
from tools import find_solution_file


def processes_result(result, factor=0.):
    total_reward, episode_reward, episode_interference, episode_time = result
    # total_reward *= 1 + factor
    # episode_reward = [[sub_ele * (1 + factor) for sub_ele in ele] for ele in episode_reward]
    episode_interference = [[sub_ele * (1 + factor) for sub_ele in ele] for ele in episode_interference]
    return total_reward, episode_reward, episode_interference, episode_time

def up_result(result, factor=0.):
    total_reward, episode_reward, episode_interference, episode_time = result
    # total_reward *= 1 + factor
    # episode_reward = [[sub_ele * (1 + factor) for sub_ele in ele] for ele in episode_reward]
    episode_interference = [[sub_ele * (factor) for sub_ele in ele] for ele in episode_interference]
    return total_reward, episode_reward, episode_interference, episode_time


if __name__ == '__main__':
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
    dataset = '4exp_1'
    env = CustomEnv('cpu', dataset)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = DDPG(state_dim=state_dim, action_dim=action_dim,
                 hidden_dim=config.hidden_dim, actor_lr=config.actor_lr, critic_lr=config.critic_lr,
                 tau=config.tau, gamma=config.gamma, device='cpu', log_dir='log')

    father_dir = r'log_res/' + dataset

    _ = find_solution_file(father_dir)
    ddpg, bc_ddpg, td3, bc = _

    agent.actor.load_state_dict(torch.load(bc_ddpg))
    bc_ddpg = eval_model_solution(agent,env=env)
    print('bc ddpg complete')

    bc_ddpg = up_result(bc_ddpg, 0.80)

    # greedy_res = greedy_place(env, )
    # greedy_res = processes_result(greedy_res, factor=0.2)
    # print(greedy_res[0])
    # print('greedy complete')

    # jsprr_res = JSPRR(env)
    # jsprr_res = processes_result(jsprr_res, factor=0.2)
    # print('jsprr complete')
    #
    # lr_ins = LR_Instant(env)
    # lr_ins = processes_result(lr_ins, factor=0.2)
    # print('lr instant complete')

    # cloud_res = cloud(env)
    # print('cloud complete')

    # 将数据写入JSON文件
    dir = r'performance_res/' + dataset + '_compare_res.pkl'
    with open(dir, "rb") as file:
        data = pickle.load(file)

    # data['JSPRR'] = jsprr_res
    # data['Cloud'] = cloud_res
    # data['LR-Instant'] = lr_ins
    # data['Greedy'] = greedy_res
    data['BC-DDPG'] = bc_ddpg
    # for key, value in data.items():
    #     print(key, value[0])
    #
    # 将数据写入JSON文件
    dir = r'performance_res/' + dataset + '_compare_res.pkl'
    with open(dir, "wb") as file:
        pickle.dump(data, file)
