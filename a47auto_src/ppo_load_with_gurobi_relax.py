# --------------------------------------------------
# 文件名: ppo
# 创建时间: 2025/3/30 20:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli

from env_calculate_delay_reward import EdgeDeploymentEnv
from rr_and_local_search import gurobi_solve_relax

env = EdgeDeploymentEnv('data/test')
# -------------------------- 环境参数 --------------------------
STATE_DIM = env.state_dim  # 状态维度（如服务器资源状态、请求队列等）
NUM_SERVERS = env.servers  # 边缘服务器数量
NUM_CONTAINERS = env.containers  # 容器类型数量
# 二维0，1
ALLOC_ACTION_DIM = env.max_requests * (env.servers_number + 1)  # 分配动作维度（服务器+云端）

# -------------------------- 超参数 --------------------------
GAMMA = 0.95  # 折扣因子
LAMBDA = 0.95  # GAE参数
CLIP_EPS = 0.2  # PPO剪切阈值
LR_ACTOR = 5e-4  # 策略网络学习率
LR_CRITIC = 5e-4  # 价值网络学习率
BATCH_SIZE = 64  # 训练批次大小
EPOCHS = 5  # 每个数据集的训练轮次
MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值
C_MAX = 0.5


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
        load_probs = [torch.sigmoid(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        # 状态价值
        value = self.value_head(features)
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (env.servers_number, env.containers_number), \
            f"shape not {load_probs.shape}"

        return load_probs, value.squeeze(-1)


# -------------------------- 经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, load_action, reward, next_state, done, old_load_logp):
        """ 存储单步经验 """
        self.buffer.append((
            state, load_action, reward, next_state, done,
            old_load_logp
        ))

    def sample(self, batch_size):
        """ 随机采样批次数据 """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        # 解包为张量
        states, load_acts, rewards, next_states, dones, old_load_logps = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)).view(batch_size, -1),
            torch.FloatTensor(np.stack(load_acts)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones),
            torch.stack(old_load_logps, dim=0),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------- 优势计算函数 --------------------------
def compute_gae(rewards, values, next_values, dones, gamma=GAMMA, lambda_=LAMBDA):
    """ 计算广义优势估计 (GAE) """
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
    return torch.FloatTensor(advantages)


# -------------------------- 训练流程 --------------------------
class PPO:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam([
            {'params': self.net.shared_layer.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.load_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.value_head.parameters(), 'lr': LR_CRITIC}
        ])
        # self.lambda_param = torch.tensor(0.1, requires_grad=True)  # 拉格朗日乘数
        # self.lambda_optim = optim.Adam([self.lambda_param], lr=1e-5)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            load_probs, value = self.net(torch.FloatTensor(state))
            # 分配动作采样
            # probs表示已经经过归一化
            # 加载动作采样
            load_dist = Bernoulli(probs=load_probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action)
            load_logp = torch.sum(load_logp, dim=-1)
        return (load_action, load_logp, value.item())

    def update(self):
        """ 执行PPO策略更新 """
        if len(self.buffer) < BATCH_SIZE:
            return

        for _ in range(EPOCHS):
            # 采样批次数据
            states, load_acts, rewards, next_states, dones, old_load_logps = \
                self.buffer.sample(BATCH_SIZE)

            assert states.shape == (BATCH_SIZE, env.state_dim), \
                f"shape not {states.shape}"
            assert load_acts.shape == (BATCH_SIZE, env.servers_number, env.containers_number), \
                f"shape not {states.shape}"

            # 计算新策略的值和动作概率
            load_probs, values = self.net(states)
            next_load_probs, next_values = self.net(next_states)
            # 加载动作对数概率
            load_dist = Bernoulli(probs=load_probs)
            new_load_logps = load_dist.log_prob(load_acts).sum(dim=-1)

            # 重要性采样比率
            ratios_load = torch.exp(new_load_logps - old_load_logps)
            ratios_load = torch.sum(ratios_load, dim=-1)
            ratios = ratios_load

            # 计算优势函数
            advantages = compute_gae(rewards.numpy(), values.detach().numpy(),
                                     next_values.detach().numpy(), dones.numpy())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 策略损失 (PPO-CLIP)
            assert ratios.shape == advantages.shape, f'not shape ratios{ratios.shape} advantage{advantages.shape}'
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            returns = advantages + values.detach()
            value_loss = F.mse_loss(values, returns)

            # 熵正则化
            entropy_load = load_dist.entropy().mean()
            entropy_loss = -0.01 * (entropy_load)

            # 总损失
            total_loss = policy_loss + value_loss + entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            return


# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(2000):
        total_reward = 0
        total_ppo_reward = 0
        invalid_actions = 0
        invalid_load_actions = 0
        compute_reward = 0
        prop_reward = 0
        state, done = env.reset()
        while not done:
            # 选择动作
            load_act, load_logp, value = agent.select_action(state)
            assert load_act.shape == (env.servers_number, env.containers_number), \
                f"not shape {load_act.shape}"
            assign_act = gurobi_solve_relax(y=load_act,
                                            requests=env.current_requests,
                                            servers=env.servers,
                                            containers=env.containers,
                                            type_map=env.service_type_to_int)
            assign_act = torch.tensor(assign_act)
            action = (assign_act, load_act)
            next_state, reward, done, info = env.step(action)
            invalid_actions += info['invalid_actions']
            compute_reward += info['compute_reward']
            prop_reward += info['prop_reward']
            # 存储经验
            agent.buffer.add(state, load_act, reward, next_state, done, load_logp)
            total_reward += reward
        agent.update()
        print("episode", episode, 'reward', total_reward)
        print("invalid load actions", invalid_load_actions)
        print("compute reward", compute_reward, 'prop_reward', prop_reward)
