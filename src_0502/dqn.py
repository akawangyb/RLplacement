# --------------------------------------------------
# 文件名: dqn
# 创建时间: 2024/5/6 12:26
# 描述: 在动作较少的情况下独立dqn算法
# 作者: WangYuanbo
# --------------------------------------------------
import random

import numpy as np
import torch
import torch.nn.functional as F

from env import CustomEnv
from memory.ReplayBuffer import ReplayBuffer
from tools import train_off_policy_agent


# import rl_utils

class Qnet(torch.nn.Module):
    """
    只有一层隐藏层的Q网络
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    """ DQN算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        # states, actions, rewards, next_states, dones = batch

        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).squeeze(dim=1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).squeeze(dim=1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # print(states)
        # print(actions.shape)
        # print(states.shape)
        # print(self.q_net(states).shape)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1




lr = 2e-3
num_episodes = 10000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# env_name = 'CartPole-v1'
# env = gym.make(env_name)
env = CustomEnv(device=device)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.state_dim
placing_action_dim = 2 ** (env.server_number * env.container_number)
routing_action_dim = (env.server_number + 1) ** env.user_number
action_dim = placing_action_dim * routing_action_dim

# print(action_dim)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
#
train_off_policy_agent(env, agent, num_episodes=num_episodes,
                       replay_buffer=replay_buffer,
                       minimal_size=minimal_size,
                       batch_size=batch_size,
                       )
