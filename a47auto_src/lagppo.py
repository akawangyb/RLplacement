# --------------------------------------------------
# 文件名: lagppo
# 创建时间: 2025/4/4 9:12
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
class PolicyNetwork(nn.Module):
    """策略网络：输出动作概率分布"""
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 加载动作头（二值）
        self.load_layer = [nn.Linear(hidden_dim, env.containers_number)
                           for _ in range(env.servers_number)]
        self.load_head = torch.nn.ModuleList(self.load_layer)

    def forward(self, x):
        features = self.fc(x)
        load_probs = [torch.sigmoid(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        return load_probs


class ValueNetwork(nn.Module):
    """价值网络：估计状态的价值 V(s)（用于奖励）"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.fc(x)


# -------------------------- LagPPO 算法类 --------------------------
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
            torch.stack(constraints, dim=0)
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        pass


# -------------------------- 训练流程 --------------------------
class PPO:
    def __init__(self):
        # 初始化网络
        self.policy_net = PolicyNetwork(env.state_dim, )
        self.value_net = ValueNetwork(env.state_dim)

        # 策略和价值网络优化器
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=LR_ACTOR)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=LR_CRITIC)

        # 初始化拉格朗日乘子（每个约束对应一个）
        # self.lag_multiplier = nn.Parameter(torch.tensor(lag_multiplier), requires_grad=True)
        self.lag_multiplier = nn.Parameter(
            torch.tensor([lag_multiplier]*env.servers_number),
            requires_grad=True)
        self.multiplier_lr = multiplier_lr  # 乘子学习率
        self.multiplier_optim = optim.Adam([self.lag_multiplier], lr=self.multiplier_lr)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):

        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            load_probs = self.policy_net(torch.FloatTensor(state))
            value = self.value_net(torch.FloatTensor(state))
            # 分配动作采样
            # probs表示已经经过归一化
            # 加载动作采样
            load_dist = Bernoulli(probs=load_probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action).sum(dim=(-2, -1))

        return (load_action, load_logp, value.item())

    def update(self):
        """ 执行PPO策略更新 """
        if len(self.buffer) < BATCH_SIZE:
            return
        # 采样批次数据
        states, load_acts, rewards, next_states, dones, old_load_logps, costs = \
            self.buffer.sample(BATCH_SIZE)
        assert states.shape == (BATCH_SIZE, env.state_dim), \
            f"shape not {states.shape}"
        assert load_acts.shape == (BATCH_SIZE, env.servers_number, env.containers_number), \
            f"shape not {states.shape}"

        for _ in range(5):
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            # 计算 GAE 和成本 GAE（使用环境提供的成本）
            advantages, returns = self.compute_gae(rewards=rewards, values=values,
                                                   next_values=next_values, dones=dones)
            # 计算平均约束违反量
            avg_cost = costs.mean()
            constraint_violation = torch.relu(avg_cost - C_MAX)  # COST_MAX为约束阈值

            new_probs = self.policy_net(states)
            dist = Bernoulli(new_probs)
            new_log_probs = dist.log_prob(load_acts).sum(dim=(-2, -1))

            # ------------ 策略损失计算（PPO Clip）------------
            ratio = (new_log_probs - old_load_logps).exp()  # 重要性采样比率
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            # 拉格朗日惩罚项
            policy_loss = -torch.min(surr1, surr2).mean() + torch.sum(self.lag_multiplier * constraint_violation)
            # ------------ 价值损失计算 ------------
            value_loss = nn.MSELoss()(values, torch.Tensor(returns))
            # entropy = dist.entropy().mean()

            # ------------ 总损失反向传播 ------------
            assert new_log_probs.shape == old_load_logps.shape, \
                f"new_log_probs shape{new_log_probs.shape} and old_load_logps shape {old_load_logps.shape}"

            # 反向传播
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            # total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy  # 0.5为价值损失系数
            total_loss = policy_loss + 0.5 * value_loss  # 0.5为价值损失系数

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
            self.optimizer_policy.step()
            self.optimizer_value.step()

            # 更新拉格朗日乘子
            lambda_loss = -torch.sum(self.lag_multiplier * constraint_violation)
            self.multiplier_optim.zero_grad()
            lambda_loss.backward()
            self.multiplier_optim.step()
            self.lag_multiplier.data.clamp_(min=0, max=10)
            # if avg_cost <= C_MAX:
            #     self.buffer.clear()  # 约束满足后清空历史数据
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
        return torch.FloatTensor(advantages), returns

# -------------------------- 训练循环示例 --------------------------
if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(config.EPOCHS):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = []
        state, done = env.reset()
        while not done:
            # 选择动作
            # 请你结合代码，讲解一下LAGPPO算法的原理
            load_act, load_logp, value = agent.select_action(state)
            assert load_act.shape == (env.servers_number, env.containers_number), \
                f"not shape {load_act.shape}"
            # assert load_act.shape == load_logp.shape, \
            #     f"not shape {load_logp.shape}"
            # alloc_act = solve_the_alloc_act(mask_load_act, env.current_requests)
            # 根据load动作计算alloc动作
            # 模拟环境返回奖励和新状态
            action = load_act
            constraint_violation = torch.Tensor(env.compute_constraint_violation(action)).mean()
            next_state, reward, done, info = env.step(action)
            # 排除全0的动作
            invalid_load_actions.append(info['invalid_load'])
            # 存储经验
            agent.buffer.add(state=state, load_action=load_act,
                             reward=reward, next_state=next_state,
                             done=done, old_load_logp=load_logp,
                             constraints=constraint_violation)
            total_reward += reward
        agent.update()
        print("episode", episode, 'reward', total_reward)
        print("invalid load actions", invalid_load_actions)
        print("lag multiplier", agent.lag_multiplier)
