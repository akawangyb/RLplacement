# --------------------------------------------------
# 文件名: agent
# 创建时间: 2024/2/12 0:56
# 描述: 智能体执行步骤的代码，实现一个简单的dqn版本
# 作者: WangYuanbo
# --------------------------------------------------
# 使用一些能够更好地处理大规模动作空间的算法，比如Dueling DQN、Prioritized Experience Replay等。

import collections
import os
import random
from collections import deque, namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from tqdm import tqdm

from src.placingENV import CustomEnv

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
    # def __init__(self, state_dim, action_dim, hidden_dim):
    #     super(Qnet, self).__init__()
    #     self.layer1 = nn.Linear(state_dim, hidden_dim)
    #     self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.layer3 = nn.Linear(hidden_dim, action_dim)
    #
    # def forward(self, x):
    #     x = F.relu(self.layer1(x))
    #     x = F.relu(self.layer2(x))
    #     return self.layer3(x)
    # 定义服务部署的动作空间
    def __init__(self, state, placing_action, routing_action, fc1_units=128,
                 fc2_units=64, hidden_dim=128):
        super(Qnet, self).__init__()
        # self.action1_size_x = action1_size_x
        # self.action1_size_y = action1_size_y
        # self.action2_size_x = action2_size_x
        # self.action2_size_y = action2_size_y
        self.placing_action = placing_action
        self.routing_action = routing_action
        self.fc1 = nn.Linear(hidden_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # 创建两个独立的全连接层，每个全连接层对应一种动作空间
        self.fc3_action1 = nn.Linear(fc2_units, placing_action[0] * placing_action[1])
        self.fc3_action2 = nn.Linear(fc2_units, routing_action[0] * routing_action[1])
        self.input_shape1 = np.prod(state['last_container_placement'].shape)
        self.input_shape2 = np.prod(state['last_request_routing'].shape)
        self.input_shape3 = np.prod(state['user_request'].shape)
        # input_shape1 = state['last_container_placement'].shape
        # input_shape2 = state['last_request_routing'].shape
        # input_shape3 = state['user_request'].shape

        # 定义一个多输入的神经网络
        # input_shape1是一个扁平化后的输入结果
        self.net_last_placing = nn.Sequential(
            nn.Linear(self.input_shape1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.net_last_routing = nn.Sequential(
            nn.Linear(self.input_shape2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.net_now_requesting = nn.Sequential(
            nn.Linear(self.input_shape3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # 定义连接网络
        self.fc = nn.Linear(128 * 3, 128)

    def forward(self, state):
        x1 = torch.Tensor(state['last_container_placement'].flatten()).view(-1, self.input_shape1)
        x2 = torch.Tensor(state['last_request_routing'].flatten()).view(-1, self.input_shape2)
        x3 = torch.Tensor(state['user_request'].flatten()).view(-1, self.input_shape3)

        x1 = self.net_last_placing(x1)
        x2 = self.net_last_routing(x2)
        x3 = self.net_now_requesting(x3)

        # 连接三个网络的输出
        x = torch.cat((x1, x2, x3), dim=0)
        x = x.view(-1, 3 * 128)
        x = F.relu(self.fc(x))
        # 到这里x是128*1
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))  # 使用sigmoid激活函数，因为我们的输出是0和1
        placing_action_tensor = self.fc3_action1(x).view(-1, self.placing_action[0], self.placing_action[1])
        routing_action_tensor = self.fc3_action2(x).view(-1, self.routing_action[0], self.routing_action[1])
        # placing_action_binarized = (placing_action_tensor > 0.5).float()
        # routing_action_binarized = (routing_action_tensor > 0.5).float()
        # placing_action_np = placing_action_binarized.numpy()
        # routing_action_np = routing_action_binarized.numpy()
        # placing_action_np_2d = np.squeeze(placing_action_np, axis=0)
        # routing_action_np_2d = np.squeeze(routing_action_np, axis=0)
        # action_dict = {'now_container_place': placing_action_np_2d, 'now_request_routing': routing_action_np_2d}
        action_dict = {'now_container_place': placing_action_tensor, 'now_request_routing': routing_action_tensor}
        # return x.view(-1, self.action_size_x, self.action_size_y)  # 改变输出层的形状以匹配动作矩阵的形状
        return action_dict


def np_to_dict(arr):
    states = {}
    for small_dict in arr:
        for key, value in small_dict.items():
            if key not in states:
                states[key] = []
            states[key].append(value)
    states = {key: np.vstack(value) for key, value in states.items()}
    return states


# 实现DQN算法
class DQN:
    def __init__(self, state, placing_action, routing_action, hidden_dim, learning_rate, gamma, epsilon,
                 target_update_interval,
                 device):
        self.state = state
        self.placing_action = placing_action
        self.routing_action = routing_action
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
        # self.q_net = Qnet(state_dim, hidden_dim,
        #                   self.action_dim).to(device)  # Q网络
        #############这里需要修改
        self.q_net = Qnet(state=state, placing_action=placing_action,
                          routing_action=routing_action).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state=state, placing_action=placing_action,
                                 routing_action=routing_action).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    # epsilon-greedy策略选择动作
    # 根据当前的状态和动作选择一个最优的做法
    def take_action(self, state):
        # 以epsilon的概率随机选择一个动作去探索未知空间
        if random.random() < self.epsilon:
            # print('探索')
            # 从动作空间随机选择动作
            # action = random.randrange(self.action_dim)
            action = env.action_space.sample()
        # 以1-epsilon概率利用已知的最优行动
        else:
            # 这里的state是一个dict,如何映射成一维数组?
            # print('开发')
            # last_container_placement_flat = state['last_container_placement'].flatten()
            # last_request_routing_flat = state['last_request_routing'].flatten()
            # user_request_flat = np.array(state['user_request'])  # 已经是一维的了，无需要铺平
            #
            # # 所有的部分拼接为一个大的一维向量
            # state = np.concatenate([last_container_placement_flat, last_request_routing_flat, user_request_flat])
            # state = torch.tensor(state, dtype=torch.float).to(self.device)
            #
            # action = self.q_net(state).argmax().item()
            action_tensor = self.q_net(state)
            action = action_tensor_to_np(action_tensor)

        # # 测试,强制使其利用
        # # 这里的state是一个dict,如何映射成一维数组?
        # last_container_placement_flat = state['last_container_placement'].flatten()
        # last_request_routing_flat = state['last_request_routing'].flatten()
        # user_request_flat = np.array(state['user_request'])  # 已经是一维的了，无需要铺平
        #
        # # 所有的部分拼接为一个大的一维向量
        # state = np.concatenate([last_container_placement_flat, last_request_routing_flat, user_request_flat])
        # state = torch.tensor(state, dtype=torch.float).to(self.device)
        # #
        # action = self.q_net(state).argmax().item()
        return action

    # 定义网络更新规则
    def update(self, transition_dict):

        # transition_dict['states']这里取出的的states是64*dict类型的,所以不能之间转成tensor

        # states = torch.tensor(transition_dict['states'],
        #                       dtype=torch.float).to(self.device)
        raw_states = transition_dict['states']
        states = np_to_dict(raw_states)
        # state是一个数组中的dict，变成一个dict中的数组
        # print(states['last_container_placement'].shape)
        # 实际上与action无关
        # raw_actions = transition_dict['actions']
        # actions = np_to_dict(raw_actions)

        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        # print(rewards.shape)
        raw_next_states = transition_dict['next_states']
        next_states = np_to_dict(raw_next_states)
        # print(next_states['last_container_placement'].shape)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # # 思考以下这个环境里面如何计算q值？
        # 原版代码
        # q_values = self.q_net(states).gather(1, actions)  # Q值
        # # 下个状态的最大Q值
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
        #     -1, 1)
        #
        # q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
        #                                                         )  # TD误差目标
        # dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # print("bug")
        # ##########################这里有问题
        q_values = self.q_net(states)['now_container_place']  # Q值
        q_values = q_values.sum(dim=(1, 2), keepdim=True).view(-1, 1)
        # print("bug1")
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states)['now_container_place']

        max_next_q_values = max_next_q_values.sum(dim=(1, 2), keepdim=True).view(-1, 1)
        # 查看各个元素的形状
        # print(rewards.shape, max_next_q_values.shape, dones.shape)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def action_tensor_to_np(action):
    placing_action_tensor = action['now_container_place']
    routing_action_tensor = action['now_request_routing']
    placing_action_binarized = (placing_action_tensor > 0.5).float()
    routing_action_binarized = (routing_action_tensor > 0.5).float()
    placing_action_np = placing_action_binarized.numpy()
    routing_action_np = routing_action_binarized.numpy()
    placing_action_np_2d = np.squeeze(placing_action_np, axis=0)
    routing_action_np_2d = np.squeeze(routing_action_np, axis=0)
    action_dict = {'now_container_place': placing_action_np_2d, 'now_request_routing': routing_action_np_2d}
    # action_dict = {'now_container_place': placing_action_tensor, 'now_request_routing': routing_action_tensor}
    # return x.view(-1, self.action_size_x, self.action_size_y)  # 改变输出层的形状以匹配动作矩阵的形状
    return action_dict


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

father_directory = '../data'
info_file = {
    'server_info.csv': 'server_id',
    'container_info.csv': 'container_id',
    'user_request_info.csv': 'timestamp',
}
# import pandas as pd
# 从csv文件读取数据
# 修改数据帧的索引
# # CSV文件名称
# csv_file = '../data/server_info.csv'  # 请用你实际的文件名替换 'your_file.csv'
for file_name in info_file:
    csv_path = os.path.join(father_directory, file_name)
    key = info_file[file_name]
    with open(csv_path, 'r', encoding='utf-8') as f:
        data_frame = pd.read_csv(f)
        data_frame.set_index(key, inplace=True)
        data_dict = data_frame.to_dict('index')
        config[file_name.replace('.csv', '')] = data_dict
env = CustomEnv(config=config)

random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
# state_dim = env.state_dim
placing_action = env.action_space['now_container_place'].shape
routing_action = env.action_space['now_request_routing'].shape

# action_dim = env.action_dim
agent = DQN(state=env.reset(), hidden_dim=hidden_dim, placing_action=placing_action, routing_action=routing_action,
            learning_rate=lr, gamma=gamma, epsilon=epsilon,
            target_update_interval=target_update, device=device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                # state,next_state 要转换成一维变量，可以直接交给qnet处理
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
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
