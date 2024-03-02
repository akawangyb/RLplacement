# --------------------------------------------------
# 文件名: pmr_dqn_demo
# 创建时间: 2024/3/1 11:37
# 描述: 优先记忆回放的demo代码
# 作者: WangYuanbo
# --------------------------------------------------

import random
from collections import namedtuple
from copy import deepcopy
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.buffer import PrioritizedReplayBuffer
from memory.utils import device

env_name = 'CartPole-v1'
env = gym.make(env_name)

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DQN:
    def __init__(self, state_size, action_size, gamma, tau, lr, device='cpu'):
        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        # 目标网络计算当前下一状态下所有行动的Q值
        # self.target_model(next_state) 是用目标模型预测 next_state 中每个动作的 Q 值，
        # 结果是一个二维张量，形状为 (batch_size, action_size)。
        # 接下来，max(dim=1) 在每一行上取最大值，
        # 结果是每个 next_state 的最大 Q 值，形状为 (batch_size, )
        Q_next = self.target_model(next_state).max(dim=1).values
        # 下一状态的目标Q值
        Q_target = reward + self.gamma * (1 - done) * Q_next
        # 实际上Q就是计算(状态,动作)的价值
        # 假设预测结果是一个4*2的张量,设有4个状态，每个状态有2个可能的动作
        # tensor([[1.2, 1.5],
        #         [1.1, 2.2],
        #         [3.2, 2.1],
        #         [2.3, 3.4]])
        # torch.arange(len(action))生成选择的行的索引[0,1,2,3]
        # action.to(torch.long).flatten()指出选择的行内坐标
        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target) ** 2 * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)

        return loss.item(), td_error

    def save(self):
        torch.save(self.model, "agent.pkl")


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
eps = 0.8
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

dqn_model = DQN(state_size=n_observations, action_size=n_actions, gamma=GAMMA, tau=TAU, lr=LR)

# memory = ReplayMemory(10000)
memory = PrioritizedReplayBuffer(state_size=n_observations, action_size=1, buffer_size=10000)

steps_done = 0

episode_durations = []

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000


def evaluate_policy(agent, episodes=5, seed=0, env_name='CartPole-v1', ):
    eval_env = gym.make(env_name)
    returns = []
    steps_done = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = eval_env.reset(seed=seed + ep)
        steps = 0
        while not done:
            steps += 1
            state, reward, terminated, truncated, _ = eval_env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        steps_done.append(steps)
        returns.append(total_reward)
    # 计算这组回报的均值和标准差
    return np.mean(returns), np.std(returns), np.mean(steps_done)


best_reward = -np.inf
rewards_total, stds_total = [], []
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    loss_count, total_loss = 0, 0
    for t in count():

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn_model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # action,reward
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        # next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(int(done), dtype=torch.int, device=device)
        # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        memory.add((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state

        if memory.real_size >= BATCH_SIZE:
            batch, weights, tree_idxs = memory.sample(BATCH_SIZE)
            loss, td_error = dqn_model.update(batch, weights)
            memory.update_priorities(tree_idxs, td_error.numpy())
            total_loss += loss
            loss_count += 1

        if done:
            if (i_episode + 1) % 10 == 0:
                episode_durations.append(t + 1)
                # 评估模型就是在episode里面,用当前策略网络计算能够获得总回报
                mean, std, steps = evaluate_policy(dqn_model, episodes=10)

                print(
                    f"Episode: {i_episode+1}, Steps: {steps}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                if mean > best_reward:
                    best_reward = mean
                    dqn_model.save()

                rewards_total.append(mean)
                stds_total.append(std)
            break
