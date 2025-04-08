# --------------------------------------------------
# 文件名: mappo
# 创建时间: 2025/4/7 23:52
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Categorical

from env_cache_sim_MA import EdgeDeploymentEnv

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

class Actor(nn.Module):
    """单智能体策略网络（局部观测输入）"""

    def __init__(self, obs_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    集中式价值网络（全局状态输入）
    参数共享：允许Actor网络共享部分参数（如特征提取层），但保留独立输出层。
    注意力机制在Critic中引入注意力机制（Attention）区分不同智能体的贡献。
    """

    def __init__(self, state_dim, num_heads=4, hidden_dim=64):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads  # 将状态维度分到多个注意力头
        self.state_dim = state_dim
        self.query = nn.Linear(state_dim, state_dim)
        self.key = nn.Linear(state_dim, state_dim)
        self.value = nn.Linear(state_dim, state_dim)
        self.fc = nn.Linear(state_dim, 1)

    def forward(self, state):
        # return self.net(state)
        # 输入x: [batch_size, state_dim]
        x = state
        batch_size = x.shape[0]
        temp = self.query(x)
        # 分割为多头的Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)  # [batch, h, d]
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)  # [batch, h, d]
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)  # [batch, h, d]

        # 计算注意力分数（缩放点积）
        # [batch, h, d] @ [batch, d, h] -> [batch, h, h]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # 应用注意力并合并头
        out = torch.matmul(attn, v)  # [batch, h, d]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1)  # [batch, state_dim]
        return self.fc(out)


class MAPPO:
    def __init__(self, n_agents=env.servers_number, obs_dim=env.obs_dim,
                 action_dim=env.containers_number, state_dim=env.state_dim,
                 actor_lr=LR_ACTOR, critic_lr=LR_CRITIC, gamma=config.GAMMA,
                 clip_eps=CLIP_EPS, K_epochs=5):
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        # 初始化策略网络（每个智能体独立）
        self.actors = [Actor(obs_dim, action_dim) for _ in range(n_agents)]
        self.critic = Critic(state_dim=state_dim, num_heads=env.servers_number)  # 集中式价值网络

        # 优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr)
                                 for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验缓冲区
        self.buffer = {
            'obs': [[] for _ in range(n_agents)],
            'actions': [[] for _ in range(n_agents)],
            'log_probs': [[] for _ in range(n_agents)],
            'rewards': [],
            'next_obs': [[] for _ in range(n_agents)],
            'state': [],
            'next_state': [],
            'dones': []
        }

    def store_transition(self, obs, actions, log_probs, rewards,
                         next_obs, state, next_state, done):
        """存储多智能体经验"""
        for i in range(self.n_agents):
            self.buffer['obs'][i].append(obs[i])
            self.buffer['actions'][i].append(actions[i])
            self.buffer['log_probs'][i].append(log_probs[i])
            self.buffer['next_obs'][i].append(next_obs[i])
        self.buffer['rewards'].append(rewards)
        self.buffer['state'].append(state)
        self.buffer['next_state'].append(next_state)
        self.buffer['dones'].append(done)

    def compute_gae(self, rewards, values, next_values, dones):
        """
        所有智能体是合作关系
        基于共享Critic计算全局GAE
        """
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)
        last_gae = 0

        for t in reversed(range(batch_size)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_gae = delta + self.gamma * 0.95 * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def update(self):
        """集中式训练更新"""
        # 转换为Tensor
        # 转换为Tensor
        buffer = {
            'state': torch.FloatTensor(np.array(self.buffer['state'])),
            'next_state': torch.FloatTensor(np.array(self.buffer['next_state'])),
            'dones': torch.FloatTensor(self.buffer['dones']),
            'rewards': [torch.FloatTensor(r) for r in self.buffer['rewards']]
        }
        for k in ['obs', 'actions', 'log_probs', 'next_obs']:
            buffer[k] = [torch.FloatTensor(np.array(v)) for v in self.buffer[k]]
        # 计算集中式价值估计
        with torch.no_grad():
            values = self.critic(buffer['state']).squeeze()
            next_values = self.critic(buffer['next_state']).squeeze()

        # 计算全局优势函数（假设奖励为团队共享）
        team_rewards = torch.mean(torch.stack(buffer['rewards']), dim=0)  # 使用团队平均奖励
        advantages, returns = self.compute_gae(
            team_rewards.numpy(),
            values.numpy(),
            next_values.numpy(),
            buffer['dones'].numpy()
        )

        # 多步更新
        for _ in range(self.K_epochs):
            # 对每个智能体独立更新策略网络
            for i in range(self.n_agents):
                # 获取当前智能体的数据
                obs_i = buffer['obs'][i]
                actions_i = buffer['actions'][i]
                old_log_probs_i = buffer['log_probs'][i]

                # 计算新策略的概率
                new_probs = self.actors[i](obs_i)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(actions_i.squeeze())

                # 重要性采样比率
                ratios = (new_log_probs - old_log_probs_i).exp()

                # PPO-Clip损失
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 更新Actor
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

            # 更新Critic（集中式）
            current_values = self.critic(buffer['state']).squeeze()
            critic_loss = nn.MSELoss()(current_values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 清空缓冲区
        self.buffer = {
            'obs': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'log_probs': [[] for _ in range(self.n_agents)],
            'rewards': [],
            'next_obs': [[] for _ in range(self.n_agents)],
            'state': [],
            'next_state': [],
            'dones': []
        }


# 示例训练循环（假设环境接口）
if __name__ == "__main__":
    env = EdgeDeploymentEnv(dataset_dir=config.data_dir)
    # 初始化MAPPO
    mappo = MAPPO()
    # 训练循环
    for episode in range(config.EPOCHS):
        obs, state, done = env.reset()
        total_rewards = np.zeros(mappo.n_agents)
        while not done:
            # 每个智能体选择动作
            actions, log_probs = [], []
            for i in range(mappo.n_agents):
                obs_tensor = torch.FloatTensor(obs[i])
                probs = mappo.actors[i](obs_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action.item())
                log_probs.append(log_prob.item())
            real_action = torch.Tensor(actions)
            # 执行动作
            next_obs, next_state, rewards, done, info = env.step(real_action)
            # for id in range(env.servers_number):
            #     print('obs', obs[id], 'action', actions[id])
            # print('obs', obs[0], 'action', actions[0])
            # print('step finished')
            # next_obs = [next_state] * mappo.n_agents
            team_reward = np.mean(rewards)
            # 存储经验
            mappo.store_transition(obs=obs, actions=actions, log_probs=log_probs,
                                   # rewards=rewards, next_obs=next_obs,
                                   rewards=[team_reward] * mappo.n_agents, next_obs=next_obs,
                                   state=state, next_state=next_state, done=int(done))

            state = next_state
            obs = next_obs
            total_rewards += rewards

        # 更新策略
        mappo.update()
        print(f"Episode {episode}, Total Reward: {sum(total_rewards)}")
        print(f"Episode {episode}, Total Reward: {total_rewards}")
