# --------------------------------------------------
# 文件名: ppo
# 创建时间: 2025/3/30 20:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import random
from collections import namedtuple
import numpy as np
import torch
import yaml
from env_cache_sim_cal_delay_interference import EdgeDeploymentEnv
from ppo import PPO
from rr_and_local_search import gurobi_solve_not_relax

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

# -------------------------- 超参数 --------------------------
env = EdgeDeploymentEnv(config.data_dir)

# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    agent = PPO(
        state_dim=env.state_dim,
        action_dim=(env.servers_number,env.containers_number),
        clip_eps=config.CLIP_EPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        gamma=config.GAMMA,
        lambda_=config.LAMBDA,
        lr_actor=config.LR_ACTOR,
        lr_critic=config.LR_CRITIC)
    # 模拟环境交互
    for episode in range(config.EPOCHS):
    # for episode in range(1):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = 0
        state, done = env.reset()
        while not done:
            # print('ts', env.current_timestep)
            # 选择动作
            load_act, load_logp, value = agent.select_action(state)
            assert load_act.shape == (env.servers_number,), \
                f"not shape {load_act.shape}, env.servers_number {env.servers_number}"
            # 下一步是根据部署动作，计算分配动作
            y_action = np.zeros((env.servers_number, env.containers_number))
            for i in range(env.servers_number):
                y_action[i][load_act[i]] = 1
            assign_act = gurobi_solve_not_relax(y=y_action,
                                                requests=env.current_requests,
                                                servers=env.servers,
                                                containers=env.containers,
                                                type_map=env.service_type_to_int,
                                                h_c_map=env.h_c_map)
            assert assign_act.shape == (len(env.current_requests), env.servers_number + 1), \
                f"assign_act not shape {assign_act.shape}"
            action = (load_act, assign_act)
            # print('load_act', y_action)
            # print('assign_act', assign_act)
            # 模拟环境返回奖励和新状态
            next_state, reward, done, info = env.step(action)
            reward = -reward
            # 存储经验
            agent.store_transition(state=state, action=load_act, reward=-reward,
                                   next_state=next_state, done=done, log_prob=load_logp)
            total_reward += reward
            # for key, value in info.items():
            #     print(key, value)
        agent.update()
        print("episode", episode, 'reward', total_reward)
