# --------------------------------------------------
# 文件名: malagppo
# 创建时间: 2025/4/6 6:44
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Bernoulli

from env_cache_sim_MA import EdgeDeploymentEnv

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
BATCH_SIZE = config.BATCH_SIZE  # 训练批次大小
EPOCHS = config.EPOCHS  # 每个数据集的训练轮次
MAX_GRAD_NORM = config.MAX_GRAD_NORM  # 梯度裁剪阈值
# lagppo算法中cmax参数的含义是什么？
C_MAX = config.C_MAX
multiplier_lr = float(config.multiplier_lr)
lag_multiplier = float(config.lag_multiplier)

env = EdgeDeploymentEnv(config.data_dir)
# -------------------------- 环境参数 --------------------------
STATE_DIM = env.state_dim  # 状态维度（如服务器资源状态、请求队列等）
NUM_SERVERS = env.servers  # 边缘服务器数量
NUM_CONTAINERS = env.containers  # 容器类型数量
# 二维0，1
ALLOC_ACTION_DIM = env.max_requests * (env.servers_number + 1)  # 分配动作维度（服务器+云端）

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# --------------------- 多智能体策略网络 ---------------------
class AgentPolicy(nn.Module):
    """单个服务器Agent的策略网络"""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, log_prob, constraints):
        """ 存储单步经验 """
        self.buffer.append((
            state, action, reward, next_state, done,
            log_prob, constraints
        ))

    def sample(self, batch_size):
        """ 随机采样批次数据 """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        # 解包为张量
        states, load_acts, rewards, next_states, dones, old_load_logps, constraints = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.stack(load_acts)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones),
            torch.stack(old_load_logps, dim=0),
            torch.FloatTensor(constraints)
        )

    # s,a,r,s',dones,logps,cost

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        pass


# --------------------- MALAGPPO算法 ---------------------
class MALAGPPO:
    def __init__(self, n_servers=env.servers_number,
                 obs_dim=env.state_dim,
                 action_dim=env.containers_number,
                 lambda_lr=multiplier_lr):
        self.n_servers = n_servers
        self.agents = [AgentPolicy(obs_dim, action_dim) for _ in range(n_servers)]
        self.lambdas = nn.Parameter(torch.ones(n_servers), requires_grad=True)
        # self.lag_multiplier = nn.Parameter(torch.ones(n_servers), requires_grad=True)

        # 优化器
        # self.agent_optims = [optim.Adam(agent.parameters(), lr=LR_ACTOR) for agent in self.agents]

        # 优化器
        # 为每个智能体的Actor和Critic分别创建优化器
        self.actor_optimizers = [
            optim.Adam(agent.actor.parameters(), lr=LR_ACTOR)
            for agent in self.agents
        ]
        self.critic_optimizers = [
            optim.Adam(agent.critic.parameters(), lr=LR_CRITIC)
            for agent in self.agents
        ]
        self.lambda_optim = optim.Adam([self.lambdas], lr=lambda_lr)

        # 超参数
        self.gamma = config.GAMMA
        self.clip_eps = config.CLIP_EPS
        self.K_epochs = 5
        self.buffer = ReplayBuffer(capacity=10000)

    def select_actions(self, state):
        """分布式决策：每个Agent基于全局状态选择动作"""
        actions, log_probs, values = [], [], []
        for agent in self.agents:
            probs, value = agent(torch.FloatTensor(state))
            dist = Bernoulli(probs)
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action).sum(dim=-1))
            values.append(value)
        # return actions, torch.stack(log_probs).unsqueeze(0), torch.stack(values).unsqueeze(0)
        return actions, torch.stack(log_probs), torch.stack(values)

    def update(self):
        """集中式训练：协同更新所有Agent的策略和lambda"""
        if len(self.buffer) < BATCH_SIZE:
            return
        # s,a,r,s',dones,logps, cost
        states, actions, rewards, next_states, dones, old_log_probs, costs = self.buffer.sample(BATCH_SIZE)
        assert rewards.shape == costs.shape, f'rewards. shape={rewards.shape}, costs.shape={costs.shape}'
        # 分离输入数据（关键步骤）
        states = states.detach().requires_grad_(False)  # 禁用梯度追踪
        actions = actions.detach()
        old_log_probs = old_log_probs.detach()
        next_states = next_states.detach().requires_grad_(False)
        # 计算全局优势估计
        with torch.no_grad():
            values = torch.stack([agent.critic(states) for agent in self.agents], dim=1).squeeze()
            # values = torch.stack([agent.critic(states) for agent in self.agents], dim=1).squeeze(dim=-1)
            next_values = torch.stack([agent.critic(next_states) for agent in self.agents], dim=1).squeeze()
            # next_values = torch.stack([agent.critic(next_states) for agent in self.agents], dim=1).squeeze(dim=-1)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # 多步优化
        for _ in range(self.K_epochs):
            total_loss = 0
            agent_losses = []
            for i, agent in enumerate(self.agents):
                current_states = states.clone()
                # 新策略的对数概率
                # probs, new_value = agent(states)
                probs, new_value = agent(current_states)
                dist = Bernoulli(probs)
                new_log_probs = dist.log_prob(actions[:, i]).sum(dim=-1)

                # PPO-Clip损失
                ratio = (new_log_probs - old_log_probs[:, i]).exp()
                surr1 = ratio * advantages[:, i]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[:, i]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                critic_loss = nn.MSELoss()(new_value.squeeze(), returns[:, i])

                # 约束损失（本地lambda）
                # mem_usage = self._compute_memory_usage(i, actions[:, i])
                constraint_violation = torch.relu(costs[:, i].mean() - C_MAX)
                constraint_loss = self.lambdas[i] * constraint_violation.mean()

                # 全局均衡损失
                util_rates = costs[:, i].mean()
                imbalance_loss = 0.1 * torch.var(util_rates, dim=-1).mean()


                # 总损失
                # agent_losses[i] = policy_loss + 0.5 * critic_loss + constraint_loss
                total_loss = policy_loss + 0.5 * critic_loss + constraint_loss + imbalance_loss
                agent_losses.append(total_loss)

            # ------------------------- 独立梯度更新 -------------------------
            # 逐个智能体反向传播并更新参数
            # 分步反向传播（隔离计算图）
            for i, loss in enumerate(agent_losses):
                self.actor_optimizers[i].zero_grad()
                self.critic_optimizers[i].zero_grad()
                # loss.backward(retain_graph=(i < len(self.agents) - 1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.agents[i].critic.parameters(), 0.5)
                self.actor_optimizers[i].step()
                self.critic_optimizers[i].step()

            # ------------------------- 更新拉格朗日乘子 -------------------------
            lambda_loss = -torch.sum(self.lambdas * torch.relu(costs.mean(dim=0) - C_MAX))
            self.lambda_optim.zero_grad()
            lambda_loss.backward()
            self.lambda_optim.step()
            self.lambdas.data = torch.clamp(self.lambdas.data, min=0.0)

        # self.buffer.clear()

    def compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计（多Agent版）"""
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(self.n_servers)
        assert rewards.shape == values.shape,\
            f"rewards shape {rewards.shape}, values shape {values.shape}"
        for t in reversed(range(len(rewards))):
            for i in range(self.n_servers):
                if dones[t]:
                    delta = rewards[t, i] - values[t, i]
                    last_gae[i] = delta
                else:
                    delta = rewards[t, i] + self.gamma * next_values[t, i] - values[t, i]
                    last_gae[i] = delta + self.gamma * 0.95 * last_gae[i]
                advantages[t, i] = last_gae[i]
        returns = advantages + values.numpy()
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)


if __name__ == '__main__':
    agent = MALAGPPO()
    # 模拟环境交互
    for episode in range(config.EPOCHS):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = []
        state, done = env.reset()
        while not done:
            # 选择动作
            actions, log_probs, values = agent.select_actions(state)
            constraint_violation = env.compute_constraint_violation(actions)
            next_state, reward, done, info = env.step(actions)
            # 排除全0的动作
            invalid_load_actions.append(info['invalid_load'])
            # 存储经验
            agent.buffer.add(state=state, action=actions,
                             reward=reward, next_state=next_state,
                             done=done, log_prob=log_probs,
                             constraints=constraint_violation)
            total_reward += sum(reward)
        agent.update()
        print("episode", episode, 'reward', total_reward)
        print("invalid load actions", invalid_load_actions)
        print("lag multiplier", agent.lambdas)
