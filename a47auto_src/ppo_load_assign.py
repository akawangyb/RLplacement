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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.distributions import Categorical

from env_cache_sim_cal_delay import EdgeDeploymentEnv
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
GAMMA = config.GAMMA  # 折扣因子
LAMBDA = config.LAMBDA  # GAE参数
CLIP_EPS = config.CLIP_EPS  # PPO剪切阈值
LR_ACTOR = float(config.LR_ACTOR)  # 策略网络学习率
LR_CRITIC = float(config.LR_CRITIC)  # 价值网络学习率
# BATCH_SIZE = config.BATCH_SIZE  # 训练批次大小
EPOCHS = config.EPOCHS  # 每个数据集的训练轮次
MAX_GRAD_NORM = config.MAX_GRAD_NORM  # 梯度裁剪阈值
C_MAX = config.C_MAX

env = EdgeDeploymentEnv(config.data_dir)
# -------------------------- 环境参数 --------------------------
STATE_DIM = env.state_dim  # 状态维度（如服务器资源状态、请求队列等）
NUM_SERVERS = env.servers  # 边缘服务器数量
NUM_CONTAINERS = env.containers  # 容器类型数量
# 二维0，1
ALLOC_ACTION_DIM = env.max_requests * (env.servers_number + 1)  # 分配动作维度（服务器+云端）


# -------------------------- 神经网络定义 --------------------------
class ActorCritic(nn.Module):
    """ 策略-价值联合网络，处理多离散动作空间 """

    def __init__(self):
        super().__init__()
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 加载动作头（二值）
        self.load_layer = [nn.Linear(128, env.containers_number)
                           for _ in range(env.servers_number)]
        self.load_head = torch.nn.ModuleList(self.load_layer)
        # 价值函数头
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, ):
        # 共享特征
        features = self.shared_layer(x)
        # 加载动作概率（应用Mask）
        load_probs = [F.softmax(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        # 状态价值
        value = self.value_head(features)
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (env.servers_number, env.containers_number), \
            f"shape not {load_probs.shape}"
        return load_probs, value.squeeze(-1)


# -------------------------- 优势计算函数 --------------------------


# -------------------------- 训练流程 --------------------------
class PPO:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam([
            {'params': self.net.shared_layer.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.load_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.value_head.parameters(), 'lr': LR_CRITIC}
        ])
        # self.buffer = ReplayBuffer(capacity=10000)
        self.buffer = []  # 经验缓冲区

    def select_action(self, state):
        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            probs, value = self.net(torch.FloatTensor(state))
            # 分配动作采样
            # probs表示已经经过归一化
            # 加载动作采样
            load_dist = Categorical(probs=probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action)
            load_logp = torch.sum(load_logp, dim=-1)
        return (load_action, load_logp, value.item())

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储单步经验"""
        self.buffer.append((
            torch.FloatTensor(state),
            action,
            torch.FloatTensor([log_prob]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        ))

    def update(self):
        """ 执行PPO策略更新 """

        # 采样批次数据
        # 解压经验数据
        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.buffer)
        batch_size = len(self.buffer)
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.cat(old_log_probs)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)

        assert states.shape == (batch_size, env.state_dim), \
            f"shape not {states.shape}, expected batch_size{batch_size}, states_dim {env.state_dim}"
        assert actions.shape == (batch_size, env.servers_number), \
            f"shape not {states.shape}"

        # 计算价值估计
        with torch.no_grad():
            probs, values = self.net(states)
            _probs, next_values = self.net(next_states)

        # 计算优势函数
        advantages, returns = self.compute_gae(rewards=rewards.numpy(), values=values.detach().numpy(),
                                               next_values=next_values.detach().numpy(),
                                               dones=dones.numpy())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # K步优化
        for _ in range(5):
            # 重新计算新策略的动作概率
            probs, current_values = self.net(states)
            new_action, new_log_probs, values = self.select_action(state)

            # 重要性采样比率
            ratios = (new_log_probs - old_log_probs).exp()

            # PPO-Clip损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            critic_loss = nn.MSELoss()(current_values, returns)

            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            self.buffer.clear()
            return

    def compute_gae(self, rewards, values, dones, next_values,
                    gamma=GAMMA, lambda_=LAMBDA):
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)
        last_gae = 0
        for t in reversed(range(batch_size)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + gamma * next_values[t] - values[t]
                last_gae = delta + gamma * lambda_ * last_gae
            advantages[t] = last_gae
        returns = [advantages[i] + values[i] for i, ele in enumerate(advantages)]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(config.EPOCHS):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = 0
        state, done = env.reset()
        while not done:
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
        agent.update()
        print("episode", episode, 'reward', total_reward)
