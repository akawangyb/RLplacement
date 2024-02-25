# --------------------------------------------------
# 文件名: placingAgent
# 创建时间: 2024/2/24 16:01
# 描述: 完成服务部署任务的智能体
# 作者: WangYuanbo
# --------------------------------------------------
# --------------------------------------------------
# S个容器要部署到N个服务器上，动作空间是2^(N*S)个，
# 考虑这样一个子问题，容器s是否部署到服务器n上，动作空间只有2个

import collections
import math
import os
import random
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from container_placing_env import CustomEnv

# 经验是一个具名元组
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# 首先定义经验回放池
class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = np.array(state)
        return state, action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class ReplayMemory(object):
    # memory实现了一个队列
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]


# 定义一个Q网络
# Q网络的作用是输入状态，输出动作
class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cpu'):
        super(Qnet, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实现DQN算法
class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon,
                 target_update_interval,
                 device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # 目标网络的更新频率
        self.target_update_interval = target_update_interval
        self.device = device

        # 记录更新次数
        self.count = 0
        # # 定义Q网络和目标Q网络
        self.q_net = Qnet(state_dim=state_dim, action_dim=action_dim, device=self.device).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim=state_dim, action_dim=action_dim, device=self.device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    # epsilon-greedy策略选择动作
    # 根据当前的状态和动作选择一个最优的做法
    def take_action(self, state):
        # 以epsilon的概率随机选择一个动作去探索未知空间
        flag = 'explore'
        global steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if random.random() < eps_threshold:
            # print('探索')
            # 从动作空间随机选择动作
            action = random.randrange(self.action_dim)
        # 以1-epsilon概率利用已知的最优行动
        else:
            # print('开发')
            # print(state)
            # print(type(state))
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state)
            action = action.argmax().item()

        return action

    # 定义网络更新规则
    def update(self, transition_dict):

        # transition_dict['states']这里取出的的states是64*dict类型的,所以不能之间转成tensor

        states = transition_dict['states']
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        actions = transition_dict['actions']
        actions = torch.tensor(actions).view(-1, 1).to(self.device)

        next_states = transition_dict['next_states']
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        # writer.add_scalar('dqn_loss', dqn_loss.item())
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


# 日志输出路径
father_log_directory = '../log'
if not os.path.exists(father_log_directory):
    os.makedirs(father_log_directory)
current_time = datetime.now()
formatted_time = current_time.strftime('%Y%m%d-%H_%M_%S')

log_path = os.path.join(father_log_directory, formatted_time)
# 规范化文件路径
log_dir = os.path.normpath(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'info.log')
f = open(log_path, 'w', encoding='utf-8')
writer = SummaryWriter(log_dir)

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
steps_done = 0

lr = 2e-3
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.5
target_update = 10
buffer_size = 10000
minimal_size = 100
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

server_storage_size = 100000
container_info = []
file_path = '../data/container_info.csv'
key = 'container_id'
data_frame = pd.read_csv(file_path)
data_frame.set_index(key, inplace=True)
data_dict = data_frame.to_dict('index')
for key in data_dict:
    container_info.append(data_dict[key]['container_size'])
penalty = -10
env = CustomEnv(server_storage_size=server_storage_size, container_info=container_info, penalty=penalty)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# print(state_dim)
# print(action_dim)
agent = DQN(state_dim, action_dim, hidden_dim, lr, gamma, epsilon,
            target_update, device)
s_sum = sum(env.container_info)
print(s_sum)

return_list = []
for i_episode in range(num_episodes):
    episode_return = 0
    state = env.reset()
    done = False
    index = 0
    while not done:
        # print(index)
        index += 1
        action = agent.take_action(state)
        # print(type(action))
        # print(action)
        # print(action.shape)
        next_state, reward, done, _ = env.step(action)
        # print('bug')
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        # 当buffer数据的数量超过一定值后,才进行Q网络训练
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            agent.update(transition_dict)
    return_list.append(episode_return)
    print(episode_return)
