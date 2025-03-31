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
from torch.distributions import Categorical, Bernoulli

from env_joint_place_distribution import EdgeDeploymentEnv

env = EdgeDeploymentEnv('../data/exptest')
# -------------------------- 环境参数 --------------------------
STATE_DIM = env.state_dim  # 状态维度（如服务器资源状态、请求队列等）
NUM_SERVERS = env.servers  # 边缘服务器数量
NUM_CONTAINERS = env.containers  # 容器类型数量
# 二维0，1
ALLOC_ACTION_DIM = env.max_requests * (env.servers_number + 1)  # 分配动作维度（服务器+云端）
# 二维0，1
LOAD_ACTION_DIM = env.servers_number * env.containers_number  # 加载动作维度

# -------------------------- 超参数 --------------------------
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE参数
CLIP_EPS = 0.2  # PPO剪切阈值
LR_ACTOR = 3e-4  # 策略网络学习率
LR_CRITIC = 1e-3  # 价值网络学习率
BATCH_SIZE = 64  # 训练批次大小
EPOCHS = 10  # 每个数据集的训练轮次
MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值


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

        # 分配动作头（离散）
        self.alloc_layer = [nn.Linear(128, env.servers_number + 1) for _ in range(env.max_requests)]
        self.alloc_head = torch.nn.ModuleList(self.alloc_layer)

        # 加载动作头（二值）
        # self.load_head = nn.Linear(128, LOAD_ACTION_DIM)
        self.load_layer = [nn.Linear(128, env.containers_number)
                           for _ in range(env.servers_number)]
        self.load_head = torch.nn.ModuleList(self.load_layer)
        # self.load_head = nn.Linear(128, LOAD_ACTION_DIM)

        # 价值函数头
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, alloc_mask=None, load_mask=None):
        # 共享特征
        features = self.shared_layer(x)

        # 分配动作logits,应用Mask
        # alloc_logits = self.alloc_head(features)
        alloc_logits = [F.softmax(alloc(features)) for alloc in self.alloc_head]
        alloc_logits = torch.stack(alloc_logits, dim=-2)

        if alloc_mask is not None:
            assert alloc_mask.shape == alloc_logits.shape, "掩码与logits维度不匹配!"
            alloc_logits[~alloc_mask] = -1e8  # 非法动作置为极小数

        # 加载动作概率（应用Mask）
        load_probs = [torch.sigmoid(layer(features)) for layer in self.load_head]
        load_probs = torch.stack(load_probs, dim=-2)
        if load_mask is not None:
            load_probs = load_probs * load_mask.float()

        # 状态价值
        value = self.value_head(features)
        assert (alloc_logits.shape[-2], alloc_logits.shape[-1]) == (env.max_requests, env.servers_number + 1), \
            f"shape not {alloc_logits.shape}"
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (env.servers_number, env.containers_number), \
            f"shape not {load_probs.shape}"
        return alloc_logits, load_probs, value.squeeze(-1)


# -------------------------- 经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, alloc_action, load_action, reward, next_state, done, old_alloc_logp, old_load_logp):
        """ 存储单步经验 """
        self.buffer.append((
            state, alloc_action, load_action, reward, next_state, done,
            old_alloc_logp, old_load_logp
        ))

    def sample(self, batch_size):
        """ 随机采样批次数据 """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        # 解包为张量
        states, alloc_acts, load_acts, rewards, next_states, dones, old_alloc_logps, old_load_logps = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)).view(batch_size, -1),
            # torch.LongTensor(np.stack(alloc_acts)),
            torch.stack(alloc_acts, dim=0),
            torch.FloatTensor(np.stack(load_acts)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones),
            # torch.FloatTensor(old_alloc_logps),
            torch.stack(old_alloc_logps, dim=0),
            # torch.FloatTensor(old_load_logps)
            torch.stack(old_load_logps, dim=0)
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
            {'params': self.net.alloc_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.load_head.parameters(), 'lr': LR_ACTOR},
            {'params': self.net.value_head.parameters(), 'lr': LR_CRITIC}
        ])
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, alloc_mask, load_mask):
        """ 使用当前策略选择动作 """
        with torch.no_grad():
            # 前向传播并应用Mask
            alloc_logits, load_probs, value = self.net(
                torch.FloatTensor(state), None, None
                # torch.FloatTensor(state).unsqueeze(0), None, None
                # alloc_mask=torch.BoolTensor(alloc_mask).unsqueeze(0) ,
                # load_mask=torch.BoolTensor(load_mask).unsqueeze(0)
            )
            # 分配动作采样
            # probs表示已经经过归一化
            alloc_dist = Categorical(probs=alloc_logits)
            # alloc_dist = Bernoulli(probs=alloc_logits)

            alloc_action = alloc_dist.sample()
            alloc_logp = alloc_dist.log_prob(alloc_action)
            assert alloc_action.shape == (env.max_requests), \
                f'allo action output shape{alloc_action.shape}'

            # 加载动作采样
            load_dist = Bernoulli(probs=load_probs)
            load_action = load_dist.sample()
            load_logp = load_dist.log_prob(load_action)
            load_logp = torch.sum(load_logp, dim=-1)
        return (
            alloc_action, load_action,
            # alloc_logp.item(), load_logp.item(),
            alloc_logp, load_logp,
            value.item()
        )

    def update(self):
        """ 执行PPO策略更新 """
        if len(self.buffer) < BATCH_SIZE:
            return

        for _ in range(EPOCHS):
            # 采样批次数据
            states, alloc_acts, load_acts, rewards, next_states, dones, old_alloc_logps, old_load_logps = \
                self.buffer.sample(BATCH_SIZE)

            assert states.shape == (BATCH_SIZE, env.state_dim), \
                f"shape not {states.shape}"
            assert alloc_acts.shape == (BATCH_SIZE, env.max_requests), \
                f"shape not {states.shape}"
            assert load_acts.shape == (BATCH_SIZE, env.servers_number, env.containers_number), \
                f"shape not {states.shape}"

            # 计算新策略的值和动作概率
            alloc_logits, load_probs, values = self.net(states)
            next_alloc_logits, _, next_values = self.net(next_states)

            # 分配动作对数概率
            alloc_dist = Categorical(logits=alloc_logits)
            new_alloc_logps = alloc_dist.log_prob(alloc_acts)

            # 加载动作对数概率
            load_dist = Bernoulli(probs=load_probs)
            new_load_logps = load_dist.log_prob(load_acts).sum(dim=-1)

            # 重要性采样比率
            # print('new', new_alloc_logps)
            # print('old', old_alloc_logps)
            ratios_alloc = torch.exp(new_alloc_logps - old_alloc_logps)
            ratios_load = torch.exp(new_load_logps - old_load_logps)
            ratios = torch.sum(ratios_alloc) * torch.sum(ratios_load)

            # 计算优势函数
            advantages = compute_gae(rewards.numpy(), values.detach().numpy(),
                                     next_values.detach().numpy(), dones.numpy())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 策略损失 (PPO-CLIP)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            returns = advantages + values.detach()
            value_loss = F.mse_loss(values, returns)

            # 熵正则化
            entropy_alloc = alloc_dist.entropy().mean()
            entropy_load = load_dist.entropy().mean()
            entropy_loss = -0.01 * (entropy_alloc + entropy_load)

            # 总损失
            total_loss = policy_loss + value_loss + entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()


# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    agent = PPO()
    # 模拟环境交互
    for episode in range(10000):
        total_reward = 0
        state, done = env.reset()
        while not done:
            alloc_mask = env.get_alloc_mask()
            load_mask = env.get_loading_mask()
            # 选择动作
            alloc_act, load_act, alloc_logp, load_logp, value = agent.select_action(
                state, alloc_mask, load_mask
            )
            # 模拟环境返回奖励和新状态
            # alloc_act = alloc_act
            action = (alloc_act, load_act)
            assert alloc_act.shape == env.max_requests, f"shape not {alloc_act.shape}"
            assert load_act.shape == (env.servers_number, env.containers_number), f"shape not {load_act.shape}"
            next_state, reward, done, _ = env.step(action)
            # for key, v in _.items():
            #     print(key, v)
            # 存储经验
            agent.buffer.add(state, alloc_act, load_act, reward, next_state, done,
                             alloc_logp, load_logp)
            total_reward += reward
        agent.update()
        print("episode", episode, 'reward', total_reward)
