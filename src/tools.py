# --------------------------------------------------
# 文件名: tools
# 创建时间: 2024/2/26 15:30
# 描述: 公共工具类
# 作者: WangYuanbo
# --------------------------------------------------
import collections
import random
from collections import deque, namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 经验是一个具名元组
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def get_user_request_info(user_number):
    user_request_info = []
    path = '../data/user_request_info.csv'
    key = 'timestamp'
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        for index in range(user_number):
            ts_request.append({
                'cpu': value['user_' + str(index) + '_cpu'],
                'imageID': int(value['user_' + str(index) + '_image']),
            })
        user_request_info.append(ts_request)
    return user_request_info


def get_server_info():
    server_info = []
    path = '../data/server_info.csv'
    key = 'server_id'
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        # print(key, value)
        server_info.append({
            'cpu': value['cpu_size'],
            'storage': value['storage_size']
        })
    return server_info


def get_container_info():
    container_info = []
    path = '../data/container_info.csv'
    key = 'container_id'
    data_frame = pd.read_csv(path)
    data_frame.set_index(key, inplace=True)
    data_dict = data_frame.to_dict('index')
    for ts, value in data_dict.items():
        # 第一个key是ts
        ts_request = []
        container_info.append({
            'pulling': value['container_pulling_delay'],
            'storage': value['container_size']
        })
    return container_info


def get_placing_info():
    path = '../data/placing_info.csv'
    key = 'server_id'
    df = pd.read_csv(path, index_col=False)
    # 删除指定列
    df = df.drop(df.columns[0], axis=1)
    placing_info = df.values.tolist()

    return placing_info


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


def onehot_from_logits(logits, eps=0.2):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(
        torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],
        requires_grad=False).to(logits.device)

    # for i, r in enumerate(torch.rand(logits.shape[0])):
    #     print(i, r)
    # print(argmax_acs.shape, rand_acs.shape)
    # print(argmax_acs.shape[0], argmax_acs.shape[1])
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y
