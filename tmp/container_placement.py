import csv
import sys
from collections import deque, namedtuple
from typing import Tuple, Optional
import torch
import gym
from gym import spaces, core
import numpy as np
from gym.core import ActType, ObsType
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim


def read_container_data():
    with open('../data/random/container.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            container_list.append([int(row[0]), int(row[1]), int(row[2])])


# 处理好模型的输入输出
# 模型的输入包含5个节点的资源余留信息，以及等待放置的容器的资源请求信息
# 可以将输入设置为 2*5，2*1
# 先实现一个最简单的版本1*12，然后再来升级
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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


def select_action(state):
    global steps_done
    sample = random.random()
    # 设置动态的epsilon有利于在前期多做探索，后期多做开发
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # 执行利用
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print("********************select*********action***********************")
            # print("policy_net(state): ", policy_net(state))
            # print("policy_net(state).max(1): ", policy_net(state).max(0))
            # print("maxQ(s,a): ", policy_net(state).max(0)[1])

            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    print(type(transitions))
    # 将具名元组的列表从列表形式的序列，转化到以元组形式的存储序列

    batch = Transition(*zip(*transitions))

    # 计算最终状态掩码，最终状态是指非结束状态
    # map()的功能是一个迭代器，有点类似一个筛子
    # 筛出batch.next_state 的非空项，返回True，None 项返回False
    # map() 的返回值是一个迭代器，可以用Tuple()强制转为元组对象
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    # print("non_final_mask: ", non_final_mask.shape)

    # 计算非最终状态下一状态
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    # print("non_final_next_states: ", non_final_next_states.shape)
    # [128,12]
    state_batch = torch.cat(batch.state)
    # [128,1]
    action_batch = torch.cat(batch.action)
    # [128]
    reward_batch = torch.cat(batch.reward)

    print("state_batch: ", state_batch.shape, "action_batch: ", action_batch.shape, "reward_batch: ",
          reward_batch.shape)

    # 计算Q(s_t,a) 输入状态计算每个动作对应的Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(policy_net(state_batch).shape)

    # 计算max_Q(s',a')
    # torch.zeros(batch_size),返回的是一个行向量
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print("next_state_values.shape: ", next_state_values.shape)
    # next_state_values[non_final_mask]这个状态掩码表示只更新非终止状态
    # max(dim)表示沿着dim维度取最大值,[0]表示只留下值，丢掉索引
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # print("next_state_values.shape: ", next_state_values.shape)
    # next_state_values 和non_final_mask分别是两个tensor，后者类型是bool,next_state_values[non_final_mask]表示什么

    # 计算TD目标  r+[gamma* max_Q(s',a')]
    # [gamma* max_Q(s',a')]表示在下一个状态s'下，选择Q值最大的动作a'执行
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算损失函数值
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 将梯度清零
    optimizer.zero_grad()
    loss.backward()
    # 裁剪梯度，防止梯度爆炸，使梯度值在[-100,100]范围内
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def draw_reward():
    # 创建一些示例数据

    # 绘制图表
    plt.plot(episode_sum)

    # 设置纵坐标的刻度值
    # plt.yticks([0, 2, 4, 6, 8, 10])

    # 显示图表
    plt.show()


class container_placement_env(gym.Env):
    def __init__(self):
        # 定义观察空间
        # 此处的观察空间包含5个节点余下的资源量，以及当前应该放入的容器需要的资源量
        self.node_low_state = np.array([0, 0])
        self.node_high_state = np.array([128, 64])

        self.container_low_state = np.array([1, 1])
        self.container_high_state = np.array([20, 4])

        self.node_space = spaces.Box(low=self.node_low_state, high=self.node_high_state, dtype=int)

        self.container_space = spaces.Box(low=self.container_low_state, high=self.node_high_state, dtype=int)

        self.observation_space = spaces.Dict({
            'node': spaces.Tuple([self.node_space] * node_number),
            'container': self.container_space
        })

        # 定义动作空间
        # 有6个动作 0-4对应放到相应的节点，5对应不放
        self.action_space = spaces.Discrete(node_number + 1)

        # 定义时间戳，表示当前来到的容器部署请求
        # self.timestamp = 0

        # 定义环境的初始状态，初始状态就是每个节点的状态都是满的
        # cpu = container_list[self.timestamp]
        # mem = container_list[self.timestamp]

        # 时间才是系统的重要状态，它决定了容器到来的请求的集合
        self.state = {'node': [node_max_cpu, node_max_mem] * node_number, 'timestamp': 0}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # 传入一个动作，根据系统当前的状态，输出下一个观察空间
        # 至于这个动作如何选择在其他模块实现
        # 要返回5个变量，分别是 state, reward, terminated, truncated, info
        self.state = self.state
        reward = 0
        terminated = False
        truncated = False
        info = {}
        # 计算奖励
        signal = self.judge(action)
        if signal == -1:
            reward = Penalty
            # self.state['timestamp'] += 1
        elif signal == 0:
            reward = 0
            # self.state['timestamp'] += 1
        else:
            # 取出放置当前容器可以获得的奖励
            reward = container_list[self.state['timestamp']][2]
            # 更新系统状态,节点的容量相应减少
            self.state = self.update(action)
        # 如果剩下的资源不多了，那么直接终止
        # 如果明显是一个不好的episode，那么直接截断
        terminated = self.check()
        self.state['timestamp'] += 1
        return self.state, reward, terminated, truncated, info
        # 如果执行了一个非法的动作，不执行这个动作，并讲奖励设置为负数。

    def reset(self):
        # 返回初始状态，每一个节点都是资源都是满的
        # self.timestamp = 0
        cpu = container_list[0][0]
        mem = container_list[0][1]
        self.state = {'node': [node_max_cpu, node_max_mem] * node_number, 'timestamp': 0}
        return self.state

    def judge(self, action: ActType) -> int:
        if action == node_number:
            return 0
        node_index = action * 2
        timestamp = self.state['timestamp']
        container_cpu = container_list[timestamp][0]
        container_mem = container_list[timestamp][1]
        node_cpu = self.state['node'][node_index]
        node_mem = self.state['node'][node_index + 1]

        if container_cpu <= node_cpu and container_mem <= node_mem:
            return 1
        else:
            return -1

    def update(self, action: ActType) -> ObsType:
        node_index = action * 2
        timestamp = self.state['timestamp']
        container_cpu = container_list[timestamp][0]
        container_mem = container_list[timestamp][1]

        self.state['node'][node_index] -= container_cpu
        self.state['node'][node_index + 1] -= container_mem

        return self.state

    def check(self) -> bool:
        arr = self.state['node']
        timestamp = self.state['timestamp']
        if timestamp == end_time - 1:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            return True
        for i in range(0, len(arr), 2):
            res_cpu = arr[i]
            res_mem = arr[i + 1]
            if res_cpu >= node_min_cpu and res_mem >= node_min_mem:
                return False
        return True


# 实验参数设置
# 对于服务器 ，假设都是同构的，cpu为128，mem为64，共 5个
# 对于一个容器，随机生成一些数据 cpu为[1,20],mem为[1,4] 共有200个
node_max_cpu = 128
node_max_mem = 64
node_min_cpu = 8
node_min_mem = 4
node_number = 5
# 先把数据全读出来然后存到变量container_list里面
container_list = []
Penalty = -50

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_observations = (node_number + 1) * 2
n_actions = node_number + 1
TAU = 0.005
steps_done = 0
end_time = 200
num_episode = 5 if device == 'cpu' else 500

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
env = container_placement_env()
memory = ReplayMemory(10000)
BATCH_SIZE = 128
LR = 1e-4
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
GAMMA = 0.99

episode_show = []

if __name__ == '__main__':
    read_container_data()
    # print("policyNet: ", policy_net)
    with open('output_data_11_29.txt', 'w', newline='') as f:
        sys.stdout = f
        for i_episode in range(num_episode):
            episode_sum = 0
            state = env.reset()
            # print("initial_state:", state)
            node_state = state['node']
            node_state = torch.tensor(node_state, dtype=torch.float32, device=device)

            container_state = [container_list[state['timestamp']][0], container_list[state['timestamp']][1]]
            container_state = torch.tensor(container_state, dtype=torch.float32, device=device)

            state = torch.cat([node_state, container_state]).unsqueeze(0)
            # print("initial_node_state: ", node_state, ",container_state: ", container_state, ",state: ", state)
            # state = torch.tensor(state, dtype=torch.float32, device=device)
            for _ in range(200):
                # 最后来实现动作如何选择的问题
                # 把状态传入策略网络中输出策略
                # action = env.action_space.sample()
                # state 是一个Dict 字典类型，要把它转为神经网络的输入，tensor类型
                # print("System************Input***************NO: ", _)
                # print("input_node_state: ", node_state, ",input_container_state: ", container_state, ",state: ", state)
                action = select_action(state)
                # print("action.item(): ", action.item())

                observation, reward, terminated, truncated, info = env.step(action.item())
                episode_sum += reward
                print("System*************Return****************************NO: ", _)
                print("Action:", action, "Observation:", observation, "Reward:", reward, "terminated:", terminated,
                      "truncated:",
                      terminated,
                      "info:", info)
                # 把系统返回的observation转化成状态输入
                node_state = observation['node']
                node_state = torch.tensor(node_state, dtype=torch.float32, device=device)
                if terminated:
                    container_state = torch.zeros(2, device=device)
                else:
                    container_state = [container_list[observation['timestamp']][0],
                                       container_list[observation['timestamp']][1]]
                    container_state = torch.tensor(container_state, dtype=torch.float32, device=device)

                # state = torch.cat([node_state, container_state])

                # 把state,action, next_state, reward转化成一组记忆存入memory里面
                # 放入memory之前需要将这些数据变成张量类型
                reward = torch.tensor([reward], device=device)
                if terminated:
                    next_state = None
                else:
                    next_state = torch.cat([node_state, container_state]).unsqueeze(0)

                memory.push(state, action, next_state, reward)

                state = next_state
                # 优化策略网络
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                done = terminated or truncated
                if done:
                    episode_show.append(episode_sum)
                    print("episode_sum: ", episode_sum)
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break

    # # 把memory输出检查一下
    # print(memory[0].state)
    # print(memory[1].state)
    # # transitions = memory.sample(BATCH_SIZE)
    # batch = Transition(*zip(*memory))
    # print(batch.state)
    # state_batch = torch.cat(batch.state, dim=0).view(5,12)
    # # print(batch)
    # print(state_batch)

    # draw_reward()
    env.close()
