# --------------------------------------------------
# 文件名: ppo
# 创建时间: 2025/3/30 20:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Bernoulli

from env_cache_sim import EdgeDeploymentEnv

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

        self.cost_head = nn.Linear(128, 1)

    def forward(self, x, ):
        # 共享特征
        features = self.shared_layer(x)
        # 加载动作概率（应用Mask）
        load_probs = [torch.sigmoid(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        # 状态价值
        value = self.value_head(features)
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (env.servers_number, env.containers_number), \
            f"shape not {load_probs.shape}"

        cost = self.cost_head(features)

        return load_probs, value.squeeze(-1), cost.squeeze(-1)


# -------------------------- 经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, load_action, reward, next_state, done, old_load_logp, constraints):
        """ 存储单步经验 """
        self.buffer.append((
            state, load_action, reward, next_state, done,
            old_load_logp, constraints
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
            torch.FloatTensor(np.stack(old_load_logps)),
            # torch.stack(old_load_logps, dim=0),
            # torch.Tensor(constraints)
            torch.stack(constraints, dim=0)
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------- 训练流程 --------------------------
class PPO:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam([
            {'params': self.net.shared_layer.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.load_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.value_head.parameters(), 'lr': LR_CRITIC}
        ])
        # 初始化拉格朗日乘子（每个约束对应一个）
        self.lag_multiplier = nn.Parameter(torch.tensor(lag_multiplier), requires_grad=True)
        self.multiplier_lr = multiplier_lr  # 乘子学习率
        self.multiplier_optim = optim.Adam([self.lag_multiplier], lr=self.multiplier_lr)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            load_probs, value, cost = self.net(torch.FloatTensor(state))
            # 分配动作采样
            # probs表示已经经过归一化
            # 加载动作采样
            load_dist = Bernoulli(probs=load_probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action)
            # load_logp = torch.mean(load_logp, dim=-1)
        return (load_action, load_logp, value.item())

    def update(self):
        """ 执行PPO策略更新 """
        if len(self.buffer) < BATCH_SIZE:
            return
        for _ in range(EPOCHS):
            # 采样批次数据
            states, load_acts, rewards, next_states, dones, old_load_logps, costs = \
                self.buffer.sample(BATCH_SIZE)
            assert load_acts.shape == old_load_logps.shape, \
                f"load act shape {load_acts.shape}, old load logps {old_load_logps.shape}"

            assert states.shape == (BATCH_SIZE, env.state_dim), \
                f"shape not {states.shape}"
            assert load_acts.shape == (BATCH_SIZE, env.servers_number, env.containers_number), \
                f"shape not {states.shape}"

            # 计算价值网络和成本网络的预测
            _, values, cost_values = self.net(states)
            _, next_values, next_cost_values = self.net(next_states)
            # 计算 GAE 和成本 GAE（使用环境提供的成本）
            gae, cost_gae = self.compute_gae(rewards=rewards, values=values,
                                             next_values=next_values,
                                             costs=costs, cost_values=cost_values,
                                             dones=dones, next_cost_values=next_cost_values)
            gae = torch.FloatTensor(gae)
            cost_gae = torch.FloatTensor(cost_gae)

            # 策略损失（PPO Clip）
            new_probs, values, _ = self.net(states)

            dist = Bernoulli(new_probs)
            new_log_probs = dist.log_prob(load_acts).sum(dim=(-2,-1))
            old_load_logps = old_load_logps.sum(dim=(-2,-1))

            assert new_log_probs.shape == old_load_logps.shape, \
                f"new_log_probs shape{new_log_probs.shape} and old_load_logps shape {old_load_logps.shape}"
            ratio = (new_log_probs - old_load_logps).exp()

            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            # 对于一个PPO算法，其动作空间是n * m的0, 1空间，请问如何计算其动作的优势函数
            # gae = gae.expand(gae, ratio, 'repeat')

            policy_loss = -torch.min(ratio * gae, clipped_ratio * gae).mean()

            # 约束损失
            cost_violation = torch.mean(cost_gae) - C_MAX
            total_loss = policy_loss + self.lag_multiplier * cost_violation

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            value_loss = nn.MSELoss()(values, rewards + GAMMA * (1 - dones) * values.detach())

            cost_value_loss = nn.MSELoss()(cost_values,  costs + GAMMA * (1 - dones) * cost_values.detach())

            # self.optimizer.zero_grad()
            # value_loss.backward()
            # self.optimizer.step()
            #
            # self.optimizer.zero_grad()
            # cost_value_loss.backward()
            # self.optimizer.step()
            # 更新拉格朗日乘子
            lambda_loss = -self.lag_multiplier * cost_violation.detach()
            self.optimizer.zero_grad()
            lambda_loss.backward()
            self.multiplier_optim.step()
            self.lag_multiplier.data.clamp_(min=0)
            return

    def compute_gae(self, rewards, values, dones, costs,
                    cost_values, next_values,
                    next_cost_values,
                    gamma=GAMMA, lambda_=LAMBDA):
        """计算广义优势估计（GAE）和成本优势"""
        batch_size = len(rewards)
        gae = np.zeros(batch_size)
        cost_gae = np.zeros(batch_size)
        last_gae = 0
        last_cost_gae = 0
        # 如果有一个【64】张量，现在把他拓展成【64，1，20】的张量，每个元素的值和原来一样，应该怎么做？
        for t in reversed(range(batch_size)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + gamma * next_values[t] - values[t]
                last_gae = delta + gamma * lambda_ * last_gae
            gae[t] = last_gae

            # 成本优势计算
            cost_delta = costs[t] + (1 - dones[t]) * gamma * next_cost_values[t] - cost_values[t]
            cost_gae[t] = cost_delta + (1 - dones[t]) * gamma * 0.95 * last_cost_gae
            last_cost_gae = cost_gae[t]
        return gae, cost_gae


if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(2000):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = 0
        state, done = env.reset()
        while not done:
            # 选择动作
            # 请你结合代码，讲解一下LAGPPO算法的原理
            load_act, load_logp, value = agent.select_action(state)
            assert load_act.shape == (env.servers_number, env.containers_number), \
                f"not shape {load_act.shape}"
            assert load_act.shape == load_logp.shape, \
                f"not shape {load_logp.shape}"
            # alloc_act = solve_the_alloc_act(mask_load_act, env.current_requests)
            # 根据load动作计算alloc动作
            # 模拟环境返回奖励和新状态
            action = load_act
            constraint_violations = torch.Tensor(env.compute_constraint_violation(action)).mean()
            next_state, reward, done, info = env.step(action)
            invalid_load_actions += info['invalid_load']
            # 存储经验
            agent.buffer.add(state=state, load_action=load_act,
                             reward=reward, next_state=next_state,
                             done=done, old_load_logp=load_logp,
                             constraints=constraint_violations)
            total_reward += reward
        agent.update()
        print("episode", episode, 'reward', total_reward)
        print("invalid load actions", invalid_load_actions)
        print("lag multiplier", agent.lag_multiplier.item())
