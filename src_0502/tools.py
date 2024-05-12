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


def get_user_request_info(timestamp, user_number):
    """

    :param user_number:
    :return: 返回一个三维数组，第一维是时间戳，第二维是用户id，第三维表示资源和延迟
    """
    user_request_info = np.zeros((timestamp, user_number, 6))  # 6分别是imageID,cpu,mem,net-in,net-out,lat
    path = 'data/user_request_info.csv'
    key = 'timestamp'
    data_frame = pd.read_csv(path)

    for index, row in data_frame.iterrows():
        # 第一个key是ts
        # ts_request = []
        if index >= timestamp:
            break
        for user_id in range(user_number):
            user_request_info[index, user_id, 0] = data_frame.loc[index, 'user_' + str(user_id) + '_image']
            user_request_info[index, user_id, 1] = data_frame.loc[index, 'user_' + str(user_id) + '_cpu']
            user_request_info[index, user_id, 2] = data_frame.loc[index, 'user_' + str(user_id) + '_mem']
            user_request_info[index, user_id, 3] = data_frame.loc[index, 'user_' + str(user_id) + '_net-in']
            user_request_info[index, user_id, 4] = data_frame.loc[index, 'user_' + str(user_id) + '_net-out']
            user_request_info[index, user_id, 5] = data_frame.loc[index, 'user_' + str(user_id) + '_lat']

    return user_request_info


def get_server_info(server_number):
    """
    :return:返回一个二维数组，第一维是 服务器id，第二维表示资源
    """
    server_info = np.zeros((server_number, 5))  # 5分别是cpu,mem,net-in,net-out,storage
    path = 'data/server_info.csv'
    data_frame = pd.read_csv(path)
    for server_id, row in data_frame.iterrows():
        server_info[server_id][0] = data_frame.loc[server_id, 'cpu_size']
        server_info[server_id, 1] = data_frame.loc[server_id, 'mem_size']
        server_info[server_id, 2] = data_frame.loc[server_id, 'net-in_size']
        server_info[server_id, 3] = data_frame.loc[server_id, 'net-out_size']
        server_info[server_id, 4] = data_frame.loc[server_id, 'storage_size']
    return server_info


def get_container_info(container_number):
    """
    :return:返回一个二维数组，第一维是 容器id，第二维表示资源
    """
    container_info = np.zeros((container_number, 2))
    path = 'data/container_info.csv'
    data_frame = pd.read_csv(path)
    for container_id, row in data_frame.iterrows():
        container_info[container_id, 0] = data_frame.loc[container_id, 'container_size']
        container_info[container_id, 1] = data_frame.loc[container_id, 'container_pulling_delay']
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

    # argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs
    # # 生成随机动作,转换成独热形式
    # rand_acs = torch.autograd.Variable(
    #     torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],
    #     requires_grad=False).to(logits.device)
    #
    # # 通过epsilon-贪婪算法来选择用哪个动作
    # return torch.stack([
    #     argmax_acs[i] if r > eps else rand_acs[i]
    #     for i, r in enumerate(torch.rand(logits.shape[0]))
    # ])


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


def to_maxtrix(placing_actions, matrix_shape):
    num_list = [torch.argmax(action) for action in placing_actions]
    binary_str = [bin(n)[2:].zfill(matrix_shape[1]) for n in num_list]
    matrix = np.zeros((matrix_shape[0], matrix_shape[1]), dtype=int)
    for row_index, row in enumerate(binary_str):
        for col_index, col in enumerate(row):
            matrix[row_index][col_index] = (binary_str[row_index][col_index] == '1')
    return matrix


def to_maxtrix_tensor(placing_actions, matrix_shape):
    num_list = torch.stack([torch.argmax(action) for action in placing_actions])
    binary_str = [torch.tensor(list(map(int, bin(n)[2:].zfill(matrix_shape[1])))) for n in num_list]
    matrix = torch.zeros(matrix_shape[0], matrix_shape[1], dtype=torch.int32)
    for row_index, row in enumerate(binary_str):
        matrix[row_index] = row
    return matrix


def to_list(routing_actions):
    num_list = [torch.argmax(action) for action in routing_actions]
    return num_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    def trans_action(env, action):
        # 需要把action 转换一下 成env_action={
        # placing_action:, (server_number * container_number) 2d 0,1矩阵
        # routing_action:, (user_number) 1d ,，给个位置的值[0,server_number+1]中}
        server_number = env.server_number
        container_number = env.container_number
        user_number = env.user_number
        placing_action_dim = 2 ** (env.server_number * env.container_number)
        routing_action_dim = (env.server_number + 1) ** env.user_number
        action_dim = placing_action_dim * routing_action_dim
        placing_action = action // routing_action_dim
        routing_action = action % routing_action_dim
        binary_str = np.binary_repr(placing_action, width=server_number * container_number)
        placing_tensor = torch.tensor([int(i) for i in binary_str], dtype=torch.int32).view(server_number,
                                                                                            container_number)
        base_m_str = np.base_repr(routing_action, base=server_number + 1)
        base_m_lst = [int(char) for char in base_m_str.zfill(user_number)]
        base_m_tensor = torch.tensor(base_m_lst, dtype=torch.int32)
        routing_tensor = base_m_tensor
        env_action = {
            'placing_action': placing_tensor,
            'routing_action': routing_tensor
        }
        return env_action

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        state, done = env.reset()
        # done = False
        while not done:
            action = agent.take_action(state)
            env_action = trans_action(env, action)
            next_state, reward, done, _ = env.step(env_action)

            reward = torch.sum(reward)
            # done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:  # 每10次输出episode的返回
            print({'episode': '%d' % (i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
        return_list.append(episode_return)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


if __name__ == '__main__':
    server_info = get_server_info(server_number=3)
    container_info = get_container_info(container_number=10)
    request_info = get_user_request_info(timestamp=10, user_number=10)
    print(server_info)
    # print(container_info)
    # print(request_info)
