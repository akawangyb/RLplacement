# --------------------------------------------------
# 文件名: ppo
# 创建时间: 2025/3/30 20:46
# 描述:
# 原始的PPO，直接输出两个动作，该算法有问题
# 作者: WangYuanbo
# --------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """ 策略-价值联合网络，处理多离散动作空间 """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.action_dim = action_dim
        self.server_num, self.container_num = self.action_dim
        # 加载动作头（二值）
        self.load_layer = [nn.Linear(128, self.container_num)
                           for _ in range(self.server_num)]
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
        assert (load_probs.shape[-2], load_probs.shape[-1]) == (self.server_num, self.container_num), \
            f"shape not {load_probs.shape}"
        return load_probs, value.squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, clip_eps, max_grad_norm,
                 gamma, lambda_, lr_actor:float, lr_critic:float):
        self.net = ActorCritic(state_dim=state_dim, action_dim=action_dim)
        self.state_dim = state_dim
        self.server_num, self.container_num = action_dim
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.optimizer = optim.Adam([
            {'params': self.net.shared_layer.parameters(), 'lr': self.lr_actor},
            {'params': self.net.load_head.parameters(), 'lr': self.lr_actor},
            {'params': self.net.value_head.parameters(), 'lr': self.lr_critic}
        ])
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
        # return (load_action, load_logp, value.item())
        return (load_action, load_logp, value)

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

        assert states.shape == (batch_size, self.state_dim), \
            f"shape not {states.shape}, expected batch_size{batch_size}, states_dim {self.state_dim}"
        assert actions.shape == (batch_size, self.server_num), \
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
            new_action, new_log_probs, values = self.select_action(states)

            # 重要性采样比率
            ratios = (new_log_probs - old_log_probs).exp()

            # PPO-Clip损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            critic_loss = nn.MSELoss()(current_values, returns)

            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.buffer.clear()
            return

    def compute_gae(self, rewards, values, dones, next_values,
                    ):
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)
        last_gae = 0
        for t in reversed(range(batch_size)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_gae = delta + self.gamma * self.lambda_ * last_gae
            advantages[t] = last_gae
        returns = [advantages[i] + values[i] for i, ele in enumerate(advantages)]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
